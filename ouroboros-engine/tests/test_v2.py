"""Invariantes da v2 e testes que distinguem comportamento real de decoração."""
from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import random
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch

from ouroboros.v2.common import seed_everything, wilson_interval
from ouroboros.v2.communication import CommunicationGame, GameConfig, make_batch, train_communication
from ouroboros.v2.communication.training import load_game
from ouroboros.v2.dashboard import handler_for
from ouroboros.v2.evolution import MorphogenicBrain, MorphologyConfig
from ouroboros.v2.integrations import verify_upstream
from ouroboros.v2.programming.evaluation import evaluate
from ouroboros.v2.programming.genetics import GeneticProgrammer
from ouroboros.v2.programming.library import ToolLibrary
from ouroboros.v2.programming.tasks import affine_task, curriculum, make_task, task_from_examples
from ouroboros.v2.programming.token_model import TokenProgrammer
from ouroboros.v2.vm import Limits, Machine, Program, assemble, compile_source, disassemble
from ouroboros.v2.world import ArenaConfig, CooperativeArena, CooperativeParallelEnv


class VMTests(unittest.TestCase):
    def test_control_flow_and_unseen_numeric_inputs(self):
        for task in curriculum():
            with self.subTest(task=task.name):
                result = evaluate(task.reference_program(), task.train + task.validation + task.test)
                self.assertTrue(result.solved, result)

    def test_assembly_roundtrip_and_serialization(self):
        program = assemble("PUSH 0\nagain:\nPUSH 1\nADD\nDUP\nPUSH 3\nLT\nJNZ again\nOUTPUT\nHALT")
        self.assertEqual(program.fingerprint, assemble(disassemble(program)).fingerprint)
        self.assertEqual(program.fingerprint, Program.from_dict(program.to_dict()).fingerprint)
        result = Machine().run(program, trace=True)
        self.assertEqual(result.outputs, [3.0])
        self.assertGreater(len(result.trace), len(program.code))

    def test_heap_allocation_and_dynamic_index(self):
        program = assemble("ALLOC 2\nDUP\nPUSH 77\nWRITE\nREAD\nOUTPUT\nHALT")
        result = Machine().run(program)
        self.assertTrue(result.ok, result)
        self.assertEqual(result.outputs, [77])
        self.assertEqual(result.allocated, 18)

    def test_fuel_covers_infinite_control_flow(self):
        result = Machine(Limits(fuel=37)).run(assemble("again:\nJMP again"))
        self.assertFalse(result.ok)
        self.assertEqual(result.fault, "fuel")
        self.assertEqual(result.steps, 37)

    def test_stack_heap_input_and_number_limits(self):
        sources = {"DROP\nHALT": "stack_underflow", "PUSH 1\nPUSH 0\nDIV\nOUTPUT": "division",
                   "PUSH -1\nREAD\nOUTPUT": "address", "ALLOC 128\nOUTPUT": "heap",
                   "PUSH 1000000000000\nPUSH 2\nMUL\nOUTPUT": "number"}
        for source, fault in sources.items():
            with self.subTest(fault=fault):
                self.assertEqual(Machine().run(assemble(source)).fault, fault)
        overflow = Machine(Limits(stack=3)).run(assemble("again:\nPUSH 1\nJMP again"))
        self.assertEqual(overflow.fault, "stack_overflow")
        self.assertEqual(Machine().run(assemble("INPUT 0\nOUTPUT"), [float("nan")]).fault, "number")
        self.assertEqual(Machine().run(assemble("LEN\nOUTPUT"), array=range(200)).fault, "input")

    def test_calls_share_stack_and_respect_depth(self):
        caller = assemble("CALL 0\nPUSH 4\nADD\nOUTPUT\nHALT")
        result = Machine(tools={0: assemble("PUSH 3\nRET")}).run(caller)
        self.assertEqual(result.outputs, [7])
        recursive = assemble("CALL 0\nRET")
        result = Machine(Limits(call_depth=3), tools={0: recursive}).run(caller)
        self.assertEqual(result.fault, "call_depth")

    def test_rejects_host_operations_and_invalid_jumps(self):
        for source in ("SYSTEM 0", "PUSH nan\nOUTPUT", "JMP 999", "CALL -1", "INPUT 9"):
            with self.subTest(source=source), self.assertRaises(ValueError): assemble(source)
        for source in ("import os", "def solve(x):\n    return x.__class__", "def solve(x):\n    return __import__('os')",
                       "def solve(x):\n    return [i for i in range(5)]", "def solve(x):\n    return open('/tmp/a')"):
            with self.subTest(source=source), self.assertRaises((ValueError, SyntaxError)):
                compile_source(source)

    def test_while_cannot_escape_fuel_limit(self):
        program = compile_source("def solve(x):\n    while x > 0:\n        x += 1\n    return x")
        self.assertEqual(Machine(Limits(fuel=100)).run(program, [1]).fault, "fuel")

    def test_empty_output_is_failure(self):
        self.assertFalse(Machine().run(assemble("HALT")).ok)
        task = make_task("sum")
        self.assertLess(evaluate(assemble("HALT"), task.train).reward, 0)


class ProgrammingTests(unittest.TestCase):
    def test_new_specification_needs_no_reference_source(self):
        from ouroboros.v2.common import ROOT
        task = task_from_examples(ROOT / "data/v2/tasks/novel_affine.json")
        self.assertEqual(task.source, "")
        self.assertEqual(task.context().shape, (72,))
        self.assertTrue(evaluate(compile_source("def solve(x, y):\n    return 3*x + 2*y - 1"), task.test).solved)

    def test_examples_encoder_is_order_invariant(self):
        seed_everything(42)
        model = TokenProgrammer(hidden=32)
        context = make_task("absolute_difference").context().reshape(8, 9)
        original = model.encoder(context.reshape(1, -1))
        permuted = model.encoder(context[torch.tensor([7, 3, 1, 5, 2, 0, 6, 4])].reshape(1, -1))
        self.assertTrue(torch.allclose(original, permuted, atol=1e-6))

    def test_token_programs_are_instruction_sequences(self):
        seed_everything(42)
        model = TokenProgrammer(hidden=32)
        task = make_task("factorial")
        tokens = model.tokenizer.encode(task.reference_program())
        restored = model.tokenizer.decode(tokens)
        self.assertEqual(restored.fingerprint, task.reference_program().fingerprint)
        context = task.context().unsqueeze(0)
        prefix = torch.tensor([tokens[:-1]])
        logits = model(context, prefix)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), torch.tensor(tokens[1:]))
        loss.backward()
        self.assertGreater(model.decoder.weight_ih_l0.grad.abs().sum(), 0)
        mask = model.tokenizer.allowed("INPUT", "cpu")
        self.assertEqual(int(mask.sum()), 8)

    def test_genetic_search_constructs_unseeded_composition(self):
        task = affine_task(3, 2, -1)
        # Uma referência que falha impede um solver de usar a solução escondida.
        task.reference_program = lambda: (_ for _ in ()).throw(AssertionError("Reference leakage"))
        candidate, history = GeneticProgrammer(generations=60, seed=42).search(task)
        self.assertTrue(evaluate(candidate, task.test).solved, history[-1])
        self.assertGreater(len(candidate.code), 5)
        self.assertTrue(history[-1]["solved_training"])

    def test_all_task_splits_are_disjoint(self):
        for task in curriculum():
            sets = [{(case.inputs, case.array) for case in split} for split in (task.train, task.validation, task.test)]
            self.assertFalse(sets[0] & sets[1], task.name)
            self.assertFalse(sets[0] & sets[2], task.name)
            self.assertFalse(sets[1] & sets[2], task.name)
            for seed in (7, 99):
                public = make_task(task.name, seed, counts=(8, 0, 0))
                augmented = {(case.inputs, case.array) for case in public.train}
                self.assertFalse(augmented & (sets[1] | sets[2]), task.name)

    def test_tool_revision_validation_rollback_and_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            library = ToolLibrary(directory)
            task = make_task("sum")
            slow = assemble("INPUT 0\nINPUT 1\nADD\nNOP\nNOP\nOUTPUT\nHALT")
            self.assertTrue(library.promote(0, "sum", slow, task)["accepted"])
            self.assertFalse(library.promote(0, "sum", assemble("PUSH 0\nOUTPUT"), task)["accepted"])
            self.assertEqual(library.registry["tools"]["0"]["revision"], 1)
            self.assertTrue(library.promote(0, "sum", task.reference_program(), task, require_improvement=True)["accepted"])
            self.assertEqual(library.call(0, [19, -3]).outputs, [16])
            self.assertTrue(library.rollback(0))
            self.assertEqual(library.registry["tools"]["0"]["revision"], 1)
            with self.assertRaises(ValueError): library.promote(0, "recursive", assemble("CALL 0\nRET"), task)


class CommunicationTests(unittest.TestCase):
    def test_private_referent_and_candidate_shuffle(self):
        config = GameConfig()
        generator = torch.Generator().manual_seed(4)
        batch = make_batch(config, 2048, generator=generator)
        selected = batch.receiver_candidates[torch.arange(len(batch.target_index)), batch.target_index]
        self.assertTrue(torch.equal(selected, batch.sender_observation))
        distribution = batch.target_index.bincount(minlength=4).float() / len(batch.target_index)
        self.assertTrue(((distribution - 0.25).abs() < 0.04).all())
        self.assertEqual(batch.receiver_candidates.shape, (2048, 4, 8))

    def test_discrete_channel_has_causal_effect_and_composes(self):
        seed_everything(42)
        model = CommunicationGame(GameConfig(protocol="factorized"))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            report = train_communication(model, directory, steps=200, evaluation_trials=512)
        self.assertGreater(report["train_domain"]["intact"]["accuracy"], 0.9)
        self.assertGreater(report["heldout_combinations"]["intact"]["accuracy"], 0.9)
        self.assertLess(report["train_domain"]["muted"]["accuracy"], 0.4)
        self.assertGreater(report["train_domain"]["causal_drop_muted"], 0.5)

    def test_resume_reproduces_continuous_training(self):
        config = GameConfig(protocol="factorized", hidden=16)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            seed_everything(3)
            continuous = CommunicationGame(config)
            train_communication(continuous, root / "continuous", steps=8, seed=3, evaluation_trials=32)
            seed_everything(3)
            partial = CommunicationGame(config)
            train_communication(partial, root / "partial", steps=4, seed=3, evaluation_trials=32)
            resumed = CommunicationGame(config)
            train_communication(resumed, root / "resumed", steps=8, seed=3,
                                resume=root / "partial/communication.pt", evaluation_trials=32)
            for key, value in continuous.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.state_dict()[key]), key)


class MorphologyTests(unittest.TestCase):
    def test_true_growth_pruning_preservation_and_gradients(self):
        seed_everything(5)
        brain = MorphogenicBrain(MorphologyConfig(initial_nodes=12))
        old = brain.input_weight.detach().clone()
        old_parameter_count = sum(p.numel() for p in brain.parameters())
        event = brain.adapt(1, 0.9)
        self.assertEqual(event["after"], 15)
        self.assertGreater(sum(p.numel() for p in brain.parameters()), old_parameter_count)
        self.assertTrue(torch.equal(old, brain.input_weight[:12]))
        event = brain.adapt(0.1, 0.1)
        self.assertEqual(event["after"], 12)
        self.assertEqual(len(event["removed"]), 3)
        output = brain(torch.randn(6, 12))
        output["thrust"].sum().backward()
        self.assertIsNotNone(brain.input_weight.grad)
        self.assertTrue(torch.isfinite(brain.input_weight.grad).all())

    def test_old_state_rejected_and_checkpoint_restored(self):
        seed_everything(5)
        brain = MorphogenicBrain()
        state = brain.initial_state(4)
        brain.adapt(1, 0.9)
        with self.assertRaises(ValueError): brain(torch.randn(4, 12), state)
        brain.eval()
        inputs = torch.randn(4, 12)
        expected = brain(inputs)["thrust"]
        restored = MorphogenicBrain.restore(brain.export()).eval()
        self.assertTrue(torch.equal(expected, restored(inputs)["thrust"]))


class WorldTests(unittest.TestCase):
    def setUp(self): seed_everything(42)

    def test_receiver_has_no_private_target_observation(self):
        arena = CooperativeArena(ArenaConfig(batch=2))
        before = arena.observation()["features"].clone()
        arena.target = (arena.target + 1) % arena.game.candidates
        after = arena.observation()["features"]
        self.assertTrue(torch.equal(before[:, 1], after[:, 1]))
        self.assertFalse(torch.equal(before[:, 0], after[:, 0]))

    def test_both_grips_and_correct_keys_are_required(self):
        arena = CooperativeArena(ArenaConfig(batch=1, obstacles=0, predator_speed=0))
        arena.target[:] = 0
        arena.positions[:] = arena.objects[:, :1]
        thrust = torch.zeros(1, 2, 2)
        choices = torch.zeros(1, 2, dtype=torch.long)
        for _ in range(4): arena.step(thrust, torch.tensor([[True, False]]), choices=choices)
        self.assertFalse(bool(arena.success[0]))
        for _ in range(3): arena.step(thrust, torch.ones(1, 2, dtype=torch.bool), choices=choices)
        self.assertTrue(bool(arena.success[0]))

    def test_wrong_initial_choice_cannot_be_repaired_by_following(self):
        arena = CooperativeArena(ArenaConfig(batch=1, obstacles=0, predator_speed=0))
        arena.target[:] = 0
        arena.positions[:] = arena.objects[:, :1]
        action, grip = torch.zeros(1, 2, 2), torch.ones(1, 2, dtype=torch.bool)
        arena.step(action, grip, choices=torch.ones(1, 2, dtype=torch.long))
        for _ in range(4): arena.step(action, grip, choices=torch.zeros(1, 2, dtype=torch.long))
        self.assertFalse(bool(arena.success[0]))
        self.assertTrue((arena.claims == 1).all())

    def test_momentum_decay_and_boundary_constraints(self):
        arena = CooperativeArena(ArenaConfig(batch=2, obstacles=0))
        arena.velocity[:] = 0.5
        before = arena.velocity.clone()
        arena.step(torch.zeros(2, 2, 2), torch.zeros(2, 2, dtype=torch.bool))
        self.assertTrue((arena.velocity.abs() < before.abs()).all())
        self.assertTrue((arena.positions >= 0).all())
        self.assertTrue((arena.positions <= arena.config.size).all())

    def test_official_pettingzoo_parallel_api(self):
        from pettingzoo.test import parallel_api_test
        with redirect_stdout(io.StringIO()):
            parallel_api_test(CooperativeParallelEnv(ArenaConfig(horizon=8)), num_cycles=24)


class IntegrationTests(unittest.TestCase):
    def test_upstream_sources_are_unchanged(self):
        self.assertTrue(all(item["ok"] for item in verify_upstream()))

    def test_vm_dashboard_request_and_path_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(directory))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/state") as response:
                    self.assertIsNone(json.load(response)["report"])
                request = Request(base + "/api/vm", json.dumps({"source": "INPUT 0\nINPUT 1\nMUL\nOUTPUT", "inputs": [6, 7]}).encode(), {"Content-Type": "application/json"})
                with urlopen(request) as response: self.assertEqual(json.load(response)["outputs"], [42])
                request = Request(base + "/api/vm", json.dumps({"source": "again:\nJMP again", "fuel": 50}).encode(), {"Content-Type": "application/json"})
                with urlopen(request) as response: self.assertEqual(json.load(response)["fault"], "fuel")
                with self.assertRaises(HTTPError) as error: urlopen(base + "/../pyproject.toml")
                self.assertEqual(error.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_confidence_intervals_bound_estimates(self):
        low, high = wilson_interval(90, 100)
        self.assertLess(low, 0.9)
        self.assertGreater(high, 0.9)


if __name__ == "__main__": unittest.main()
