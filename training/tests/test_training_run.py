"""Short CPU integration check for the server training environment."""

import os
import subprocess
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


@unittest.skipUnless(find_spec("torch"), "requires the separate training environment")
class TrainingRunTest(unittest.TestCase):
    def run_cli(self, module: str, *args: str | Path) -> str:
        result = subprocess.run(
            [sys.executable, "-m", module, "--device", "cpu", *map(str, args)],
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_train_resume_evaluate(self):
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            latest = output / "latest.pt"
            self.run_cli(
                "training.train", "--output", output, "--num-envs", "2", "--steps", "8",
                "--batch-size", "2", "--warmup", "2", "--replay-size", "16",
                "--max-episode-steps", "4",
            )
            initial = torch.load(latest, map_location="cpu", weights_only=True)
            self.assertEqual(initial["steps"], 8)
            self.assertGreater(initial["updates"], 0)
            self.assertEqual(len(initial["replay"]["items"]), 8)

            self.run_cli("training.train", "--resume", latest, "--steps", "12")
            resumed = torch.load(latest, map_location="cpu", weights_only=True)
            self.assertEqual(resumed["steps"], 12)
            self.assertGreater(resumed["updates"], initial["updates"])
            self.assertEqual(len(resumed["replay"]["items"]), 12)
            self.assertEqual(resumed["replay"]["items"][:8], initial["replay"]["items"])
            self.assertTrue(any(
                not torch.equal(value, resumed["online"][key])
                for key, value in initial["online"].items()
            ))

            report = self.run_cli(
                "training.evaluate", "--checkpoint", output / "policy.pt", "--max-steps", "2",
            )
            self.assertRegex(report, r"总通关 \d+/30")
