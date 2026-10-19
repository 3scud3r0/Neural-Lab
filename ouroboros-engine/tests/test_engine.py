from http.server import ThreadingHTTPServer
import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

import torch
from ouroboros.config import ROOT, read_yaml, resolve_device, validate
from ouroboros.primitives import LiquidCell, PlasticSynapse, SpikingGate, TTTLayer
from ouroboros.environment import SharedWorld, SignalBus
from ouroboros.dashboard import handler_for
from ouroboros.sandbox import ExecutionResult, SafeExecutor
from ouroboros.sandbox.feedback_bridge import feedback_from_result
from ouroboros.simulation import simulate
from ouroboros.agents.social_agent import SocialAgent


class PrimitiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_liquid_time_and_gradient(self):
        cell = LiquidCell(3, 5)
        x, h = torch.ones(2, 3), torch.zeros(2, 5)
        short, long = cell(x, h, 0.01), cell(x, h, 0.5)
        self.assertGreater(long.abs().sum(), short.abs().sum())
        long.square().mean().backward()
        self.assertGreater(cell.target.weight.grad.abs().sum(), 0)
        with self.assertRaises(ValueError):
            cell(x, h, 0)

    def test_surrogate_and_binary_spikes(self):
        x = torch.tensor([[0.1, 0.7, -0.8]], requires_grad=True)
        y, spikes = SpikingGate()(x)
        self.assertEqual(spikes.tolist(), [[0, 1, 1]])
        y.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertNotEqual(x.grad[0, 0].item(), 0)

    def test_hebbian_memory_is_individual_and_bounded(self):
        layer = PlasticSynapse(4, 2, learning_rate=1, clip=0.1)
        x = torch.stack([torch.ones(4), torch.zeros(4)])
        layer(x, torch.ones(2))
        self.assertGreater(layer.trace[0].abs().sum(), 0)
        self.assertEqual(layer.trace[1].abs().sum(), 0)
        self.assertLessEqual(layer.trace.abs().max().item(), 0.100001)

    def test_ttt_updates_without_breaking_backprop(self):
        layer = TTTLayer(4, 2)
        x = torch.ones(2, 4, requires_grad=True)
        layer(x)
        out = layer(x * 0.5)
        self.assertGreater(layer.fast.abs().sum(), 0)
        out.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


class WorldTests(unittest.TestCase):
    def test_signal_distance_and_self_exclusion(self):
        positions = torch.tensor([[0., 0.], [1., 0.], [9., 0.]])
        signals = torch.tensor([[2.], [4.], [20.]])
        received = SignalBus(2).receive(positions, signals)
        self.assertEqual(received.tolist(), [[2.], [1.], [0.]])

    def test_resource_not_duplicated_by_competition(self):
        config = read_yaml(ROOT / "configs/ecosystem_config.yaml")["world"]
        world = SharedWorld(4, "cpu", dict(config, resource_count=1, resource_regeneration=0, obstacle_count=0))
        world.positions[:] = 5
        world.sources[:] = 5
        harvest, _ = world.step(torch.zeros(4, 2), 0.1)
        self.assertAlmostEqual(harvest.sum().item(), 1)
        self.assertAlmostEqual(world.stock.item(), 0)

    def test_boundaries_and_obstacles(self):
        config = read_yaml(ROOT / "configs/ecosystem_config.yaml")["world"]
        world = SharedWorld(1, "cpu", dict(config, obstacle_count=1))
        world.positions[:] = 5
        world.obstacle_phase[:] = 5
        before = world.positions.clone()
        world.step(torch.ones(1, 2), 0.1)
        self.assertTrue(torch.equal(before, world.positions))
        self.assertTrue((world.positions >= 0).all())


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.executor = SafeExecutor()

    def test_correct_and_wrong_program(self):
        cases = [[2, 3, 5], [-9, 4, -5]]
        good = self.executor.execute("x + y", cases)
        bad = self.executor.execute("0", cases)
        self.assertTrue(good.ok, good.message)
        self.assertTrue(feedback_from_result(good).solved)
        self.assertFalse(feedback_from_result(bad).solved)
        self.assertLess(feedback_from_result(bad).reward, feedback_from_result(good).reward)
        self.assertGreater(good.memory_kb, 0)
        self.assertLess(good.memory_kb, 256 * 1024)

    def test_rejects_python_escape_and_large_inputs(self):
        for expression in ["__import__('os').getcwd()", "x.__class__", "open('/tmp/test', 'w')",
                           "[x for x in range(10)]", "2 ** 99999", "x / 0", "'abc'", "True", "x" * 300]:
            with self.subTest(expression=expression):
                result = self.executor.execute(expression, [[1, 2, 3]])
                self.assertFalse(result.ok)
                self.assertFalse(feedback_from_result(result).solved)

    def test_timeout_and_empty_output_are_not_success(self):
        result = SafeExecutor(timeout_seconds=0.000001).execute("x+y", [[1, 2, 3]])
        self.assertFalse(result.ok)
        self.assertFalse(feedback_from_result(ExecutionResult(True, error=0)).solved)
        self.assertFalse(self.executor.execute("x", []).ok)


class IntegrationTests(unittest.TestCase):
    def test_learning_run_outputs_and_reproducibility(self):
        config = read_yaml(ROOT / "configs/ecosystem_config.yaml")
        primitives = read_yaml(ROOT / "configs/neuron_primitives.yaml")
        config.update(steps=12, population=4, hidden_size=8, device="cpu")
        config["programmer"]["interval"] = 6
        torch.manual_seed(config["seed"])
        SharedWorld(config["population"], "cpu", config["world"])
        initial = SocialAgent(config, primitives, "cpu").brain.state_dict()
        with tempfile.TemporaryDirectory() as temporary:
            config["output_dir"] = str(Path(temporary) / "first")
            first = simulate(config, primitives, ROOT / "data/bootstrap_code/tasks.json", quiet=True)
            output = Path(config["output_dir"])
            rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(math.isfinite(row["loss"]) for row in rows))
            self.assertEqual(first["programmer"]["attempts"], 2)
            self.assertEqual(len(list((output / "programs").glob("*.py"))), 2)
            saved = torch.load(output / "ecosystem.pt", weights_only=True)
            self.assertGreater(saved["brain"]["ttt.fast"].abs().sum(), 0)
            for weight in ("predictor.weight", "motor.weight", "voice.weight"):
                self.assertGreater((saved["brain"][weight] - initial[weight]).abs().sum(), 0)
            self.assertTrue(json.loads((output / "latest.json").read_text())["completed"])
            with self.assertRaises(ValueError):
                simulate(config, primitives, ROOT / "data/bootstrap_code/tasks.json", quiet=True)
            config["output_dir"] = str(Path(temporary) / "second")
            second = simulate(config, primitives, ROOT / "data/bootstrap_code/tasks.json", quiet=True)
            for key in ["energy", "loss", "reward", "deaths", "spike_fraction"]:
                self.assertEqual(first[key], second[key])

    def test_configuration_rejects_invalid_values(self):
        config = read_yaml(ROOT / "configs/ecosystem_config.yaml")
        primitives = read_yaml(ROOT / "configs/neuron_primitives.yaml")
        config["population"] = 0
        with self.assertRaises(ValueError):
            validate(config, primitives)
        if not torch.cuda.is_available():
            with self.assertRaises(RuntimeError):
                resolve_device("cuda")

    def test_dashboard_endpoints_waiting_live_and_path_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(output))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base) as response:
                    self.assertIn("Ouroboros", response.read().decode())
                with self.assertRaises(HTTPError) as error:
                    urlopen(base + "/api/latest")
                self.assertEqual(error.exception.code, 503)
                (output / "latest.json").write_text('{"step": 7}')
                (output / "metrics.jsonl").write_text('{"energy": 0.9}\n{"partial":')
                with urlopen(base + "/api/latest") as response:
                    self.assertEqual(json.load(response)["step"], 7)
                with urlopen(base + "/api/history") as response:
                    self.assertEqual(len(json.load(response)), 1)
                with self.assertRaises(HTTPError) as error:
                    urlopen(base + "/../pyproject.toml")
                self.assertEqual(error.exception.code, 404)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
