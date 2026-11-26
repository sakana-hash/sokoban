"""Short CPU integration check for the server training environment."""

import os
import random
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

    def test_train_validate_resume_evaluate(self):
        import torch
        from training.env import ACTIONS, AREA, DIRECTIONS, SIDE

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            latest = output / "latest.pt"
            logs = self.run_cli(
                "training.train", "--output", output, "--num-envs", "2", "--steps", "8",
                "--batch-size", "2", "--warmup", "2", "--replay-size", "16",
                "--max-episode-steps", "4", "--curriculum-steps", "4", "--validation-rooms", "1",
            )
            initial = torch.load(latest, map_location="cpu", weights_only=True)
            best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            self.assertIn('"validation_steps": 4', logs)
            self.assertIn('"validation_steps": 8', logs)
            self.assertEqual(initial["version"], 2)
            self.assertIn(best["steps"], (4, 8))
            self.assertEqual(initial["best_policy"]["wins"], best["wins"])
            for name, weight in best["online"].items():
                self.assertTrue(torch.equal(weight, initial["best_policy"]["online"][name]))
            self.assertEqual(initial["steps"], 8)
            self.assertGreater(initial["updates"], 0)
            experience = initial["replay"]["items"]
            self.assertEqual(len(experience), 8)
            for state, action, _, successor, mask, _ in experience:
                self.assertIn(action, range(4))
                self.assertEqual(len(mask), 4)
                # A replay step is exactly one adjacent player move.
                y0, x0 = divmod(state[3 * AREA:].index(1), SIDE)
                y1, x1 = divmod(successor[3 * AREA:].index(1), SIDE)
                self.assertEqual((x1 - x0, y1 - y0), DIRECTIONS[ACTIONS[action]])

            self.run_cli("training.train", "--resume", latest, "--steps", "12")
            resumed = torch.load(latest, map_location="cpu", weights_only=True)
            self.assertEqual(resumed["steps"], 12)
            self.assertGreater(resumed["updates"], initial["updates"])
            self.assertGreaterEqual(resumed["stage"], initial["stage"])
            self.assertGreaterEqual(resumed["best_policy"]["wins"], best["wins"])
            self.assertEqual(len(resumed["replay"]["items"]), 12)
            self.assertEqual(resumed["replay"]["items"][:8], initial["replay"]["items"])
            self.assertTrue(any(
                not torch.equal(value, resumed["online"][key])
                for key, value in initial["online"].items()
            ))

            report = self.run_cli(
                "training.evaluate", "--checkpoint", output / "best.pt", "--max-steps", "2",
                "--compare-random",
            )
            self.assertRegex(report, r"总通关 \d+/30")
            self.assertIn("随机基线", report)

    def test_v1_walk_weights_evaluate_and_full_checkpoint_resumes(self):
        import torch
        from training.dqn import QNetwork, ReplayBuffer, optimize
        from training.env import ACTIONS, Game, encode, legal_mask, step_game

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rng = random.Random(42)
            game = Game(("########", "#@   $.#", "########"))
            replay = ReplayBuffer(16)
            online, target = QNetwork(), QNetwork()
            target.load_state_dict(online.state_dict())
            optimizer = torch.optim.Adam(online.parameters(), lr=3e-4)
            for direction in ("right", "right", "left", "left"):
                state = encode(game)
                action = ACTIONS.index(direction)
                reward = step_game(game, action)
                replay.add(state, action, reward, encode(game), legal_mask(game), game.won)
            loss = optimize(
                online, target, optimizer, replay, 2, 0.99, torch.device("cpu"), rng,
            )
            # Explicit original fields: loading this checkpoint exercises missing-v2-field defaults.
            legacy = {
                "version": 1,
                "steps": 4,
                "online": online.state_dict(),
                "config": {
                    "seed": 42, "num_envs": 1, "batch_size": 2, "replay_size": 16,
                    "warmup": 2, "epsilon_steps": 150000, "curriculum_steps": 50000,
                    "max_episode_steps": 256,
                },
                "target": target.state_dict(),
                "optimizer": optimizer.state_dict(),
                "replay": replay.state_dict(),
                "updates": 1,
                "episodes": 0,
                "recent_wins": [],
                "last_loss": loss.item(),
                "slots": [{"board": game.observation()["board"], "length": 4}],
                "rng": rng.getstate(),
                "torch_rng": torch.get_rng_state(),
            }
            latest = output / "latest.pt"
            torch.save(legacy, latest)
            policy_path = output / "policy.pt"
            torch.save({key: legacy[key] for key in ("version", "steps", "online")}, policy_path)
            report = self.run_cli(
                "training.evaluate", "--checkpoint", policy_path, "--generated", "3",
                "--max-boxes", "1", "--max-steps", "2", "--compare-random",
            )
            self.assertRegex(report, r"总通关 \d+/3")
            self.run_cli("training.train", "--resume", latest, "--steps", "8")
            resumed = torch.load(latest, map_location="cpu", weights_only=True)
            QNetwork().load_state_dict(resumed["online"], strict=True)
            self.assertEqual(resumed["steps"], 8)
            self.assertEqual(resumed["replay"]["items"][:4], legacy["replay"]["items"])
            self.assertGreater(resumed["updates"], legacy["updates"])
