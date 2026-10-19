"""Behavioral checks for every one of the sixteen v3 modules."""
from contextlib import redirect_stdout
import copy
from dataclasses import asdict
import importlib
import io
import json
from pathlib import Path
import random
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen
from http.server import ThreadingHTTPServer

import numpy as np
import torch
from pettingzoo.test import parallel_api_test

from ouroboros.v3 import MODULES, Score, SearchConfig, fingerprint
from ouroboros.v3.application import ROOT, dashboard_handler, export_code, integrated_checks, run_experiment
from ouroboros.v3.architecture import HeterogeneousBrain, Region, development_decision, resize_region, search_architectures
from ouroboros.v3.benchmarks import Baseline, TASKS, evaluate, make_batch, train_steps
from ouroboros.v3.evaluation import compare_models, interval, promotion_decision
from ouroboros.v3.experiments import Budget, ExperimentStore, isolated_seed, load_checkpoint, save_checkpoint, seed_all
from ouroboros.v3.memory import Consolidation, EpisodicMemory
from ouroboros.v3.neuron_dsl import Expr, compile_expression, parse
from ouroboros.v3.neurons import InventedCell, SequenceModel, seed_expressions
from ouroboros.v3.plasticity import Plasticity
from ouroboros.v3.programming import FunctionSpec, ModulePlan, Programmer
from ouroboros.v3.research import Hypothesis, ResearchController
from ouroboros.v3.search import NeuronSearch, behavior_signature, crossover, diverse_archive, mutate, pareto, random_expression
from ouroboros.v3.selfmod import RevisionWorkspace
from ouroboros.v3.society import Population, train_free_language
from ouroboros.v3.world import ConstructionWorld, WorldConfig
from ouroboros.v2.programming.tasks import affine_task, make_task
from ouroboros.v2.vm import Machine, assemble


class ContractsTests(unittest.TestCase):
    def setUp(self):
        seed_all(42)

    def test_exactly_sixteen_modules_import(self):
        modules = sorted(path.stem for path in (ROOT / "ouroboros/v3").glob("*.py"))
        self.assertEqual(modules, sorted(MODULES))
        for name in MODULES:
            self.assertIsNotNone(importlib.import_module(f"ouroboros.v3.{name}"))

    def test_invalid_search_configuration(self):
        for values in ({"candidates": 0}, {"hidden": 129}, {"length": 129}, {"max_seconds": float("nan")}, {"batch": True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                SearchConfig(**values)

    def test_fingerprint_is_canonical(self):
        self.assertEqual(fingerprint({"a": 1, "b": 2}), fingerprint({"b": 2, "a": 1}))
        self.assertFalse(Score("x", float("nan"), 1, 0, "screen").valid)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        seed_all(42)

    def test_store_reopens_and_rejects_config_change(self):
        with tempfile.TemporaryDirectory() as folder:
            store = ExperimentStore(folder, {"seed": 42})
            store.record("neuron", "screen", {"loss": 0.5})
            store.close()
            store = ExperimentStore(folder, {"seed": 42})
            self.assertEqual(store.get("neuron", "screen"), {"loss": 0.5})
            self.assertEqual(len(store.rows("screen")), 1)
            store.close()
            with self.assertRaises(ValueError):
                ExperimentStore(folder, {"seed": 43})

    def test_budget_stops(self):
        budget = Budget(10, 1)
        budget.claim()
        with self.assertRaises(TimeoutError): budget.claim()
        expired = Budget(-1, 1)
        with self.assertRaises(TimeoutError): expired.claim()

    def test_seed_context_restores_rng(self):
        expected_torch = torch.get_rng_state().clone()
        expected_python = random.getstate()
        with isolated_seed(17):
            torch.rand(3)
            random.random()
        self.assertTrue(torch.equal(expected_torch, torch.get_rng_state()))
        self.assertEqual(expected_python, random.getstate())

    def test_optimizer_checkpoint_exact_continuation(self):
        expression = seed_expressions()["leaky"]
        model = SequenceModel(3, 5, 1, expression)
        optimizer = train_steps(model, 3, 17)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            save_checkpoint(path, model, optimizer, {"step": 3})
            train_steps(model, 2, 17, optimizer=optimizer, start=3)
            expected = copy.deepcopy(model.state_dict())
            restored = SequenceModel(3, 5, 1, expression)
            restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.015)
            metadata = load_checkpoint(path, restored, restored_optimizer)
            train_steps(restored, 2, 17, optimizer=restored_optimizer, start=3)
            self.assertEqual(metadata["step"], 3)
            for name, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, expected[name]), name)


class DSLTests(unittest.TestCase):
    def test_parse_roundtrip_and_differentiation(self):
        expression = parse({"op": "tanh", "args": [{"op": "mul", "args": [{"op": "x"}, {"op": "h"}]}]})
        self.assertEqual(expression.identity, parse(expression.to_dict()).identity)
        function = compile_expression(expression)
        x = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
        other = torch.randn(2, 3, dtype=torch.double)
        self.assertTrue(torch.autograd.gradcheck(lambda x: function(x=x, h=other, m=other, a=other, p=other), (x,)))

    def test_rejects_host_code_shapes_and_resource_exhaustion(self):
        for value in ({"op": "__import__"}, {"op": "x", "code": "x"}, {"op": "add", "args": []},
                      {"op": "const", "value": float("inf")}, {"op": "const", "value": True}):
            with self.subTest(value=value), self.assertRaises(ValueError): parse(value)
        deep = {"op": "x"}
        for _ in range(10): deep = {"op": "tanh", "args": [deep]}
        with self.assertRaises(ValueError): parse(deep)
        fn = compile_expression({"op": "x"})
        with self.assertRaises(ValueError): fn(x=torch.ones(1))
        with self.assertRaises(ValueError):
            fn(x=torch.ones(1), h=torch.ones(2), m=torch.ones(1), a=torch.ones(1), p=torch.ones(1))


class NeuronTests(unittest.TestCase):
    def test_all_families_have_finite_dynamics_and_gradients(self):
        seed_all(42)
        for name, expr in seed_expressions().items():
            with self.subTest(name=name):
                cell = InventedCell(3, 7, expr, spiking=name == "spiking")
                state = None
                for _ in range(20):
                    output, state = cell(torch.randn(4, 3), state)
                self.assertTrue(torch.isfinite(output).all())
                self.assertLessEqual(output.abs().max(), 8)
                output.square().mean().backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in cell.parameters()))

    def test_state_affects_output_and_detaches(self):
        cell = InventedCell(3, 5, seed_expressions()["leaky"])
        inputs = torch.ones(2, 3)
        first, state = cell(inputs)
        second, _ = cell(inputs, state)
        self.assertFalse(torch.equal(first, second))
        self.assertFalse(state.detach().h.requires_grad)
        with self.assertRaises(ValueError): cell(torch.ones(3))


class PlasticityTests(unittest.TestCase):
    def test_rules_change_fast_weights_and_stay_bounded(self):
        for mode in ("hebbian", "eligibility", "learned"):
            rule = Plasticity(3, 2, rate=0.1, limit=0.2)
            for _ in range(10):
                proposal = rule.update(torch.ones(4, 3), torch.ones(4, 2), mode=mode)
            self.assertGreater(rule.fast.abs().sum(), 0)
            self.assertLessEqual(rule.fast.abs().max(), 0.200001)
            self.assertEqual(rule(torch.ones(4, 3)).shape, (4, 2))
            if mode == "learned":
                proposal.sum().backward()
                self.assertTrue(any(p.grad is not None for p in rule.modulator.parameters()))
            rule.reset()
            self.assertEqual(rule.fast.abs().sum(), 0)

    def test_stdp_uses_temporal_order_and_reward_reverses_updates(self):
        rule = Plasticity(1, 1)
        rule.update(torch.ones(1, 1), torch.zeros(1, 1), mode="stdp")
        rule.update(torch.zeros(1, 1), torch.ones(1, 1), mode="stdp")
        self.assertGreater(rule.fast.item(), 0)
        positive, negative = Plasticity(1, 1), Plasticity(1, 1)
        positive.update(torch.ones(1, 1), torch.ones(1, 1), reward=1)
        negative.update(torch.ones(1, 1), torch.ones(1, 1), reward=-1)
        self.assertAlmostEqual(positive.fast.item(), -negative.fast.item())


class BenchmarkTests(unittest.TestCase):
    def test_partitions_reproducible_and_distinct(self):
        for task in TASKS:
            a = make_batch(task, "train", 42)
            b = make_batch(task, "train", 42)
            c = make_batch(task, "test", 42)
            self.assertTrue(torch.equal(a.inputs, b.inputs))
            self.assertFalse(torch.equal(a.inputs, c.inputs))

    def test_delay_targets_require_past_input(self):
        data = make_batch("delay", "test", 1)
        self.assertTrue(torch.equal(data.targets[:, 2:, 0], data.inputs[:, :-2, 0]))

    def test_baselines_train_and_evaluate(self):
        seed_all(42)
        for kind in ("mlp", "rnn", "gru", "lstm"):
            model = Baseline(kind, 6)
            old = {n: p.clone() for n, p in model.named_parameters()}
            train_steps(model, 4, 42)
            self.assertTrue(any(not torch.equal(p, old[n]) for n, p in model.named_parameters()))
            self.assertGreaterEqual(evaluate(model, "test", 42)["mean"], 0)


class SearchTests(unittest.TestCase):
    def test_mutation_and_crossover_generate_valid_distinct_equations(self):
        rng = random.Random(42)
        expression = seed_expressions()["leaky"]
        variants = set()
        for _ in range(100):
            child = crossover(mutate(expression, rng), random_expression(rng), rng)
            variants.add(parse(child.to_dict()).identity)
        self.assertGreater(len(variants), 50)
        self.assertNotEqual(behavior_signature(Expr("x")), behavior_signature(Expr("h")))
        self.assertEqual(behavior_signature(Expr("x")), behavior_signature(Expr("add", (Expr("x"), Expr("const", value=0.0)))))

    def test_pareto_and_diversity(self):
        a, b, c = Score("a", 0.1, 10, 0, "x"), Score("b", 0.2, 20, 0, "x"), Score("c", 0.3, 5, 0, "x")
        self.assertEqual({s.candidate for s in pareto([a, b, c])}, {"a", "c"})
        expressions = {"a": Expr("x"), "b": Expr("x"), "c": Expr("h")}
        self.assertEqual({s.candidate for s in diverse_archive(expressions, [a, b, c])}, {"a", "c"})

    def test_search_promotes_and_resume_reuses_observations(self):
        seed_all(42)
        with tempfile.TemporaryDirectory() as folder:
            config = SearchConfig(candidates=12, hidden=5, screening_steps=1, promotion_steps=2,
                                  finalists=2, batch=4, length=5)
            search = NeuronSearch(folder, config)
            try:
                first = search.run()
                second = search.run()
                self.assertEqual(first, second)
                self.assertEqual(first["unique_equations"], 12)
                self.assertEqual(len(search.store.rows("screen")), 12)
                self.assertEqual(first["promoted"], 2)
            finally:
                search.store.close()


class EvaluationTests(unittest.TestCase):
    def test_intervals_and_insufficient_evidence(self):
        self.assertAlmostEqual(interval([1, 2, 3])["mean"], 2)
        with self.assertRaises(ValueError): interval([])
        result = promotion_decision([{"name": "candidate", "test_interval": {"n": 1}}])
        self.assertFalse(result["accepted"])

    def test_promotion_rejects_inferior_candidate(self):
        def record(name, loss):
            return {"name": name, "test_interval": {"n": 5, "mean": loss},
                    "runs": [{"seed": i, "test": {"mean": loss}} for i in range(5)]}
        self.assertFalse(promotion_decision([record("candidate", 0.4), record("gru", 0.1)])["accepted"])
        self.assertTrue(promotion_decision([record("candidate", 0.1), record("gru", 0.4)])["accepted"])

    def test_comparison_evaluates_six_models_on_reserved_data(self):
        seed_all(42)
        records = compare_models(seed_expressions()["leaky"], 4, 1, [1001, 1002], batch=4, length=5)
        self.assertEqual(len(records), 6)
        self.assertTrue(all(r["test_interval"]["n"] == 2 for r in records))
        self.assertTrue(all("transfer" in r["runs"][0] for r in records))


class ArchitectureTests(unittest.TestCase):
    def test_joint_search_changes_topology_and_neuron_equations(self):
        seed_all(42)
        result = search_architectures(list(seed_expressions().values()), candidates=6, steps=1)
        self.assertEqual(len(result["candidates"]), 6)
        self.assertIn(result["selected"], range(6))
        self.assertGreater(len({r["parameters"] for r in result["candidates"]}), 1)
        self.assertEqual(result["selected"], min(result["candidates"], key=lambda r: r["validation"]["mean"])["index"])

    def test_growth_preserves_output_and_optimizer_moments(self):
        seed_all(42)
        expr = seed_expressions()["leaky"]
        brain = HeterogeneousBrain(3, 2, [Region("a", 4, expr), Region("b", 3, expr)])
        optimizer = torch.optim.Adam(brain.parameters())
        inputs = torch.randn(5, 3)
        output, _ = brain(inputs)
        output.square().sum().backward()
        optimizer.step()
        expected, states = brain(inputs)
        old = copy.deepcopy(optimizer.state[brain.regions["a"].input.weight]["exp_avg"])
        optimizer, migrated = resize_region(brain, "a", 7, optimizer, states)
        actual, _ = brain(inputs)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-6))
        self.assertTrue(torch.equal(optimizer.state[brain.regions["a"].input.weight]["exp_avg"][:4], old))
        self.assertTrue(torch.equal(migrated["a"].h[:, :4], states["a"].h))
        self.assertEqual(migrated["a"].h.shape, (5, 7))
        optimizer, migrated = resize_region(brain, "a", 3, optimizer, migrated)
        optimizer.zero_grad()
        output, _ = brain(inputs, migrated)
        output.square().sum().backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(output).all())

    def test_development_respects_energy_and_stress(self):
        self.assertEqual(development_decision(0.9, 0.8, 12), 13)
        self.assertEqual(development_decision(0.9, 0.1, 12), 11)
        self.assertEqual(development_decision(0.1, 0.8, 12), 12)


class MemoryTests(unittest.TestCase):
    def test_retrieval_capacity_and_priority(self):
        memory = EpisodicMemory(2)
        memory.add(torch.tensor([1.0, 0]), "first", 2)
        memory.add(torch.tensor([0.0, 1]), "second", 1)
        memory.add(torch.tensor([-1.0, 0]), "third", 3)
        self.assertEqual(len(memory.items), 2)
        self.assertEqual(memory.retrieve(torch.tensor([1.0, 0]))[0][0], "first")
        self.assertEqual(len(memory.replay(4, random.Random(42))), 4)

    def test_consolidation_penalizes_drift(self):
        model = torch.nn.Linear(2, 1)
        memory = Consolidation()
        memory.capture(model, [model(torch.ones(1, 2)).square().sum()])
        self.assertEqual(memory.penalty(model).item(), 0)
        with torch.no_grad(): model.weight.add_(0.5)
        self.assertGreater(memory.penalty(model).item(), 0)


class WorldTests(unittest.TestCase):
    def test_official_pettingzoo_parallel_contract(self):
        with redirect_stdout(io.StringIO()):
            parallel_api_test(ConstructionWorld(WorldConfig(horizon=8)), num_cycles=20)

    def test_gather_craft_build_have_real_effects(self):
        world = ConstructionWorld()
        world.reset(seed=42)
        world.positions["agent_0"] = np.array([0, 0])
        world.positions["agent_1"] = np.array([6, 6])
        world.resources[0, 0] = 8
        action = lambda physical: {"agent_0": np.array([physical, 0]), "agent_1": np.array([0, 0])}
        world.step(action(4))
        world.step(action(4))
        self.assertEqual(world.inventory["agent_0"], 2)
        world.step(action(5))
        self.assertTrue(world.tools["agent_0"])
        world.step(action(4))
        self.assertEqual(world.inventory["agent_0"], 2)
        world.step(action(4))
        _, rewards, _, _, _ = world.step(action(6))
        self.assertEqual(world.structure, 1)
        self.assertGreater(rewards["agent_1"], 0)

    def test_other_agents_private_clue_is_hidden(self):
        world = ConstructionWorld()
        world.reset(seed=42)
        before = world.observe("agent_0")
        world.clues["agent_1"] = 999
        self.assertTrue(np.array_equal(before, world.observe("agent_0")))

    def test_reset_reproducible_and_invalid_actions_rejected(self):
        world = ConstructionWorld()
        first, _ = world.reset(seed=42)
        second, _ = world.reset(seed=42)
        self.assertTrue(all(np.array_equal(first[a], second[a]) for a in first))
        with self.assertRaises(ValueError): world.step({})


class SocietyTests(unittest.TestCase):
    def test_individuals_are_independent_and_inherit(self):
        seed_all(42)
        population = Population(2, seed_expressions()["leaky"], hidden=5)
        a, b = population.members
        self.assertIsNot(a.memory, b.memory)
        self.assertNotEqual(next(a.brain.parameters()).data_ptr(), next(b.brain.parameters()).data_ptr())
        child = population.reproduce(0, mutation=0)
        self.assertEqual(child.parent, a.identity)
        self.assertEqual(child.generation, 1)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a.brain.parameters(), child.brain.parameters())))
        self.assertEqual(len(child.memory.items), 0)

    def test_world_policy_gradient_updates_and_social_distillation(self):
        seed_all(42)
        population = Population(2, seed_expressions()["leaky"], hidden=5)
        class RichWorld(ConstructionWorld):
            def reset(self, *args, **kwargs):
                observations, infos = super().reset(*args, **kwargs)
                self.resources.fill(10)
                return {a: self.observe(a) for a in self.agents}, infos

        environment = RichWorld(WorldConfig(horizon=8))
        with torch.no_grad():
            for member in population.members:
                member.brain.readout.bias[:7].fill_(-8)
                member.brain.readout.bias[4] = 8
        old_policy = [p.clone() for p in population.members[0].brain.parameters()]
        result = population.episode(environment, 42, train=True)
        self.assertEqual(result["steps"], 8)
        self.assertGreater(result["rewards"]["agent_0"], 0)
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(old_policy, population.members[0].brain.parameters())))
        self.assertGreater(len(population.members[0].memory.items), 0)
        before = [p.clone() for p in population.members[1].brain.parameters()]
        history = population.distill(0, 1, torch.randn(8, 16), steps=3)
        self.assertEqual(len(history), 3)
        self.assertTrue(any(not torch.equal(x, y) for x, y in zip(before, population.members[1].brain.parameters())))

    def test_free_language_adapter_runs_real_training(self):
        seed_all(42)
        with tempfile.TemporaryDirectory() as folder:
            report = train_free_language(Path(folder), steps=2)
            self.assertTrue((Path(folder) / "communication.pt").is_file())
            self.assertIsInstance(report, dict)


class ProgrammingTests(unittest.TestCase):
    def test_module_dependency_order_and_cycle_rejection(self):
        specs = [FunctionSpec("main", ("number",), dependencies=("helper",)), FunctionSpec("helper", ("number",))]
        self.assertEqual(ModulePlan(specs).order(), ["helper", "main"])
        with self.assertRaises(ValueError):
            ModulePlan([FunctionSpec("a", (), dependencies=("b",)), FunctionSpec("b", (), dependencies=("a",))]).order()
        with self.assertRaises(ValueError): FunctionSpec("x", ("object",))

    def test_compile_debug_promote_call_and_rollback(self):
        with tempfile.TemporaryDirectory() as folder:
            programmer, task = Programmer(folder), make_task("sum")
            result = programmer.compile_and_test("def solve(x, y):\n    return x + y", task)
            self.assertTrue(result["accepted"])
            self.assertTrue(programmer.promote(0, "sum", result["program"], task)["accepted"])
            self.assertEqual(programmer.library.call(0, [17, -9]).outputs, [8])
            self.assertEqual(len(programmer.debug(assemble("PUSH 0\nOUTPUT\nHALT"), task.validation)), len(task.validation))
            composed = programmer.compose([0])
            self.assertEqual(Machine(tools=programmer.library.callable_programs()).run(composed, [2, 3]).outputs, [5])
            self.assertTrue(programmer.library.rollback(0))

    def test_synthesis_and_existing_neural_checkpoint(self):
        from ouroboros.v2.programming.evaluation import evaluate as program_evaluate
        seed_all(42)
        with tempfile.TemporaryDirectory() as folder:
            programmer = Programmer(Path(folder) / "tools")
            task = affine_task(1, 1, 0)
            program, history = programmer.synthesize(task, generations=20)
            self.assertTrue(history)
            self.assertTrue(program_evaluate(program, task.test).solved)
            checkpoint = ROOT / "validation-v2/programming/token_programmer.pt"
            self.assertTrue(checkpoint.is_file(), str(checkpoint))
            report = programmer.generate_with_checkpoint(checkpoint, make_task("sum"), Path(folder) / "generated", samples=2)
            self.assertTrue(report["solved"], report)


class SelfModificationTests(unittest.TestCase):
    def test_rejection_acceptance_persistence_and_rollback(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = RevisionWorkspace(folder, {"minimum": 5})
            proposal = workspace.propose(seed_expressions()["leaky"])
            rejected = workspace.evaluate_and_promote(proposal, lambda expr, contract: {"accepted": False})
            self.assertFalse(rejected["accepted"])
            self.assertIsNone(workspace.state["active"])
            workspace.evaluate_and_promote(proposal, lambda expr, contract: {"accepted": True, "tests": 10})
            restored = RevisionWorkspace(folder, {"minimum": 5})
            self.assertEqual(restored.state["active"], proposal["identity"])
            self.assertTrue(restored.rollback())
            self.assertIsNone(restored.state["active"])

    def test_tampering_and_contract_changes_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            workspace = RevisionWorkspace(folder, {"minimum": 5})
            proposal = workspace.propose(seed_expressions()["leaky"])
            proposal["expression"] = {"op": "x"}
            with self.assertRaises(ValueError): workspace.evaluate_and_promote(proposal, lambda *args: {"accepted": True})
            valid = workspace.propose(seed_expressions()["rate"])
            workspace.evaluate_and_promote(valid, lambda *args: {"accepted": True})
            with self.assertRaises(ValueError): RevisionWorkspace(folder, {"minimum": 1})


class ResearchTests(unittest.TestCase):
    def test_scheduler_explores_and_spends_exact_budget(self):
        hypotheses = [Hypothesis("a", "Question A"), Hypothesis("b", "Question B")]
        controller = ResearchController(hypotheses, trials=8)
        with tempfile.TemporaryDirectory() as folder:
            report = controller.run(lambda h, trial: (1 if h.name == "a" else 0, {"trial": trial}), Path(folder) / "research.json")
            self.assertEqual(report["evaluations"], 8)
            self.assertTrue(all(h.trials > 0 for h in hypotheses))
            self.assertGreater(hypotheses[0].trials, hypotheses[1].trials)
            self.assertEqual(json.loads((Path(folder) / "research.json").read_text()), report)


class ApplicationTests(unittest.TestCase):
    def test_integrated_workflow_executes(self):
        seed_all(42)
        with tempfile.TemporaryDirectory() as folder:
            report = integrated_checks(folder, seed_expressions()["leaky"])
            self.assertTrue(report["morphology"]["finite"])
            self.assertEqual(report["programming"]["call_outputs"], [8])
            self.assertFalse(report["selfmod"]["accepted"])
            self.assertEqual(report["research"]["evaluations"], 4)

    def test_full_pipeline_selects_before_independent_test(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = SearchConfig(candidates=8, hidden=4, screening_steps=1, promotion_steps=1, finalists=2, batch=4, length=5)
            report = run_experiment(folder, cfg, seeds=2)
            self.assertEqual(report["search"]["screened"], 8)
            self.assertFalse(report["promotion"]["accepted"])
            self.assertEqual(len(report["independent_comparison"]), 6)
            self.assertTrue((Path(folder) / "report.json").is_file())

    def test_export_contains_code_without_documentation_or_test_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "ouroboros").mkdir()
            (root / "ouroboros/a.py").write_text("print('source')\n")
            (root / "ouroboros/README.md").write_text("DOCUMENTATION_MARKER")
            (root / "ouroboros/results.txt").write_text("TEST_RESULTS_MARKER")
            result = export_code(root / "code.txt", root)
            text = (root / "code.txt").read_text()
            self.assertEqual(result["files"], 1)
            self.assertIn("print('source')", text)
            self.assertNotIn("DOCUMENTATION_MARKER", text)
            self.assertNotIn("TEST_RESULTS_MARKER", text)
            with self.assertRaises(ValueError): export_code(root / "ouroboros/a.py", root)

    def test_dashboard_serves_real_report_and_escapes_html(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "report.json").write_text(json.dumps({"result": "<script>evil()</script>"}))
            server = ThreadingHTTPServer(("127.0.0.1", 0), dashboard_handler(folder))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/api/report") as response:
                    self.assertIn("result", json.load(response))
                with urlopen(base + "/") as response:
                    text = response.read().decode()
                    self.assertIn("&lt;script&gt;", text)
                    self.assertNotIn("<script>evil", text)
                with self.assertRaises(HTTPError): urlopen(base + "/missing")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
