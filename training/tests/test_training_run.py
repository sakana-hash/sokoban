"""Short CPU integration check for the server training environment."""

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch


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
            console = self.run_cli(
                "training.train", "--output", output, "--num-envs", "2", "--steps", "8",
                "--batch-size", "2", "--warmup", "2", "--replay-size", "16",
                "--max-episode-steps", "4", "--curriculum-steps", "4", "--validation-rooms", "1",
            )
            metrics = output / "metrics.jsonl"
            first_lines = metrics.read_text(encoding="utf-8").splitlines()
            start, *events = map(json.loads, first_lines)
            self.assertEqual((start["event"], start["steps"], start["target_steps"]), ("start", 0, 8))
            self.assertIsNone(start["resume"])
            self.assertEqual(start["config"]["num_envs"], 2)
            self.assertEqual([e["event"] for e in events], ["validation", "train", "validation"])
            self.assertEqual(
                [{k: v for k, v in e.items() if k not in ("event", "timestamp")} for e in events],
                [json.loads(line) for line in console.splitlines() if line.startswith("{")],
            )
            initial = torch.load(latest, map_location="cpu", weights_only=True)
            experience = initial["replay"]["items"]
            self.assertEqual(len(experience), 8)
            for state, action, _, successor, mask, _ in experience:
                self.assertIn(action, range(4))
                self.assertEqual(len(mask), 4)
                y0, x0 = divmod(state[3 * AREA:].index(1), SIDE)
                y1, x1 = divmod(successor[3 * AREA:].index(1), SIDE)
                self.assertEqual((x1 - x0, y1 - y0), DIRECTIONS[ACTIONS[action]])

            self.run_cli("training.train", "--resume", latest, "--steps", "12")
            lines = metrics.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[:len(first_lines)], first_lines)
            restart, train, validation = map(json.loads, lines[len(first_lines):])
            self.assertEqual((restart["event"], restart["steps"], restart["resume"]), ("start", 8, str(latest)))
            self.assertEqual((train["steps"], validation["validation_steps"]), (12, 12))
            resumed = torch.load(latest, map_location="cpu", weights_only=True)
            self.assertEqual(resumed["steps"], 12)
            self.assertGreater(resumed["updates"], initial["updates"])
            self.assertEqual(len(resumed["replay"]["items"]), 12)
            self.assertEqual(resumed["replay"]["items"][:8], experience)
            self.assertTrue(any(
                not torch.equal(value, resumed["online"][key])
                for key, value in initial["online"].items()
            ))

            report = self.run_cli(
                "training.evaluate", "--checkpoint", output / "best.pt", "--max-steps", "2",
            )
            self.assertRegex(report, r"总通关 \d+/30")

    def test_v1_walk_weights_evaluate_and_full_checkpoint_resumes(self):
        import torch
        from training.dqn import QNetwork, ReplayBuffer
        from training.env import ACTIONS, Game, encode, legal_mask, step_game

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            game = Game(("########", "#@   $.#", "########"))
            replay = ReplayBuffer(16)
            state = encode(game)
            action = ACTIONS.index("right")
            reward = step_game(game, action)
            replay.add(state, action, reward, encode(game), legal_mask(game), game.won)
            online = QNetwork()
            # Original v1 schema during warmup, before the first gradient update.
            legacy = {
                "version": 1,
                "steps": 1,
                "online": online.state_dict(),
                "config": {
                    "seed": 42, "num_envs": 1, "batch_size": 2, "replay_size": 16,
                    "warmup": 2, "epsilon_steps": 150000, "curriculum_steps": 50000,
                    "max_episode_steps": 256,
                },
                "target": online.state_dict(),
                "optimizer": torch.optim.Adam(online.parameters(), lr=3e-4).state_dict(),
                "replay": replay.state_dict(),
                "updates": 0,
                "episodes": 0,
                "recent_wins": [],
                "last_loss": None,
                "slots": [{"board": game.observation()["board"], "length": 1}],
                "rng": random.Random(42).getstate(),
                "torch_rng": torch.get_rng_state(),
            }
            latest = output / "latest.pt"
            torch.save(legacy, latest)
            policy_path = output / "policy.pt"
            torch.save({key: legacy[key] for key in ("version", "steps", "online")}, policy_path)
            report = self.run_cli(
                "training.evaluate", "--checkpoint", policy_path, "--generated", "1", "--max-steps", "2",
            )
            self.assertRegex(report, r"总通关 \d+/1")
            self.run_cli("training.train", "--resume", latest, "--steps", "4")
            resumed = torch.load(latest, map_location="cpu", weights_only=True)
            self.assertEqual(resumed["version"], 1)
            self.assertEqual(resumed["steps"], 4)
            self.assertGreater(resumed["updates"], 0)

    def test_resume_mid_walk_restores_the_next_revisit_reward(self):
        import torch
        from training.env import ACTIONS, Game, potential, shaped_reward
        from training.runner import DISCOUNT, Episode, Trainer
        from training.train import Config

        config = Config(num_envs=1)
        trainer = Trainer(config, "cpu", None)
        game = Game(("#######", "# @ $.#", "#######"))
        trainer.slots = [Episode(game, 2)]
        with patch.object(trainer, "actions", return_value=[ACTIONS.index("left")]):
            trainer.advance(1)
        with tempfile.TemporaryDirectory() as temporary:
            trainer.save(Path(temporary))
            checkpoint = torch.load(Path(temporary) / "latest.pt", weights_only=True)

        # At the left wall the only legal move returns to the starting square.
        for has_history, expected_base in ((True, -0.03), (False, -0.01)):
            with self.subTest(has_history=has_history):
                if not has_history:
                    del checkpoint["slots"][0]["visited_since_push"]
                resumed = Trainer(config, "cpu", checkpoint)
                game = resumed.slots[0].game
                before = potential(game)
                resumed.advance(1)
                expected = shaped_reward(expected_base, before, potential(game), False, DISCOUNT)
                self.assertAlmostEqual(resumed.replay.state_dict()["items"][-1][2], expected)

    def test_episode_counts_across_resumes(self):
        import torch
        from training.env import ACTIONS, Game
        from training.runner import Episode, Trainer
        from training.train import Config

        config = Config(num_envs=3, max_episode_steps=2)
        fields = ("episodes", "episodes_won", "episodes_lost", "episodes_unknown")

        def finish_two_games(trainer):
            corridor = ("#######", "#@  $.#", "#######")
            trainer.slots = [
                Episode(Game(("#####", "#@$.#", "#####")), 2, length=1),  # Win on the last step.
                Episode(Game(corridor), 2, length=1),  # Time limit without a win.
                Episode(Game(corridor), 2),  # Still in progress.
            ]
            with patch.object(trainer, "actions", return_value=[ACTIONS.index("right")] * 3):
                trainer.advance(3)

        trainer = Trainer(config, "cpu", None)
        finish_two_games(trainer)
        metrics = trainer.training_metrics()
        self.assertEqual(tuple(metrics[key] for key in fields), (2, 1, 1, 0))

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            trainer.save(output)
            checkpoint = torch.load(output / "latest.pt", weights_only=True)
            self.assertEqual(tuple(checkpoint[key] for key in fields), (2, 1, 1, 0))

            for key in fields[1:]:
                del checkpoint[key]
            checkpoint.update(steps=300, episodes=120, recent_wins=[True] * 60 + [False] * 40)
            legacy = Trainer(config, "cpu", checkpoint)
            finish_two_games(legacy)
            legacy.save(output)
            resumed = Trainer(config, "cpu", torch.load(output / "latest.pt", weights_only=True))
            metrics = resumed.training_metrics()
            self.assertEqual(tuple(metrics[key] for key in fields), (122, 61, 41, 20))
