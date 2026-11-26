import json
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from training.env import ACTIONS, Game
from training.evaluate import evaluation_boards, play_level, run_evaluation


class EvaluationTests(unittest.TestCase):
    def test_reports_preserve_repeated_evaluations_and_matched_baseline(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            args = Namespace(checkpoint=Path(directory) / "best.pt", generated=8, max_boxes=1,
                             seed=2027, max_steps=5, compare_random=True)
            reports = args.checkpoint.parent / "evaluations"
            def greedy(state, mask):
                return mask.index(True)

            run_evaluation(args, greedy, "cpu", 2, 230000)
            first, = reports.glob("*.json")
            paired = json.loads(first.read_text(encoding="utf-8"))
            self.assertTrue(paired["deadlock_detection"])
            self.assertEqual((paired["checkpoint"], paired["checkpoint_steps"]), ("best.pt", 230000))
            self.assertEqual([row["level"] for row in paired["levels"]], list(range(1, 9)))
            for policy, field in [("model", "wins"), ("random", "random_wins")]:
                count = sum(row[policy]["outcome"] == "通关" for row in paired["levels"])
                self.assertEqual(paired["summary"][field], count)
                self.assertEqual(paired["by_boxes"]["1"][field], count)
            for policy, field in [("model", "deadlocks"), ("random", "random_deadlocks")]:
                count = sum(row[policy]["outcome"] == "死锁" for row in paired["levels"])
                self.assertEqual(paired["summary"][field], count)
            groups = {"small": [], "large": []}
            boards = evaluation_boards(args.generated, args.max_boxes, args.seed)
            for row, (_, rows) in zip(paired["levels"], boards, strict=True):
                width, height = len(rows[0]), len(rows)
                self.assertEqual((row["width"], row["height"]), (width, height))
                groups["small" if max(width, height) <= 7 else "large"].append(row)
            for name, group in groups.items():
                self.assertEqual(paired["by_board_size"][name], {
                    "attempts": len(group), "wins": sum(row["model"]["outcome"] == "通关" for row in group),
                    "random_wins": sum(row["random"]["outcome"] == "通关" for row in group),
                })

            args.compare_random = False
            run_evaluation(args, greedy, "cpu", 2, None)
            second, = set(reports.glob("*.json")) - {first}
            solo = json.loads(second.read_text(encoding="utf-8"))
            self.assertIsNone(solo["checkpoint_steps"])
            self.assertNotIn("random_wins", solo["summary"])
            self.assertNotIn("random_deadlocks", solo["summary"])
            self.assertTrue(all("random_wins" not in group for group in solo["by_board_size"].values()))
            self.assertTrue(all("random" not in row for row in solo["levels"]))
            self.assertEqual([row["model"] for row in solo["levels"]], [row["model"] for row in paired["levels"]])
            self.assertEqual(json.loads(first.read_text(encoding="utf-8")), paired)

    def test_generated_benchmark_is_reproducible(self):
        rooms = list(evaluation_boards(10, 1, 73))
        self.assertEqual(len(rooms), 10)
        self.assertTrue(all(len(Game(rows).boxes) == 1 for _, rows in rooms))
        self.assertEqual(rooms, list(evaluation_boards(10, 1, 73)))

    def test_stochastic_revisits_can_still_win(self):
        rows = ("#######", "# @ $.#", "#######")

        def follow_directions():
            sequence = iter(["left", "right", "right", "right"])
            return lambda state, mask: ACTIONS.index(next(sequence))

        self.assertEqual(play_level(Game(rows), follow_directions(), 4), "循环")
        self.assertEqual(
            play_level(Game(rows), follow_directions(), 4, detect_cycle=False), "通关"
        )

    def test_deadlock_on_the_last_move_and_at_start(self):
        rows = ("######", "#.@$ #", "#    #", "######")
        for detect_cycle in (True, False):
            with self.subTest(detect_cycle=detect_cycle):
                game = Game(rows)
                outcome = play_level(game, lambda *_: ACTIONS.index("right"), 1, detect_cycle=detect_cycle)
                self.assertEqual((outcome, game.moves, game.pushes), ("死锁", 1, 1))
                outcome = play_level(game, lambda *_: self.fail("deadlocked game requested an action"), 5)
                self.assertEqual((outcome, game.moves), ("死锁", 1))

    def test_historical_evaluation_continues_after_deadlock_for_both_policies(self):
        rows = ("######", "#.@$ #", "#    #", "######")
        game = Game(rows)
        result = play_level(game, lambda *_: ACTIONS.index("right"), 1, deadlock_detection=False)
        self.assertEqual(result, "达到步数上限")
        deadlocked = game.observation()["board"]
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            args = Namespace(checkpoint=Path(directory) / "best.pt", generated=1, max_boxes=1,
                             seed=2027, max_steps=1, compare_random=True, deadlock_detection=False)
            with patch("training.evaluate.evaluation_boards", return_value=[("死锁开局", deadlocked)]):
                run_evaluation(args, lambda _, mask: mask.index(True), "cpu", 3, 1000)
            path, = (Path(directory) / "evaluations").glob("*.json")
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(report["deadlock_detection"])
            for policy in ("model", "random"):
                self.assertEqual(report["levels"][0][policy]["outcome"], "达到步数上限")

    def test_each_physical_move_requires_one_four_way_decision(self):
        rows = ("########", "#@  $. #", "########")
        for budget, expected in [(2, "达到步数上限"), (3, "通关")]:
            game = Game(rows)
            calls = []

            def choose_right(state, mask):
                self.assertEqual(len(mask), 4)
                calls.append(state)
                return ACTIONS.index("right")

            self.assertEqual(play_level(game, choose_right, budget), expected)
            self.assertEqual(len(calls), budget)
            self.assertEqual(game.moves, budget)
            self.assertEqual(game.pushes, int(budget == 3))
