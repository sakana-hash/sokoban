import json
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

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
            self.assertEqual((paired["checkpoint"], paired["checkpoint_steps"]), ("best.pt", 230000))
            self.assertEqual([row["level"] for row in paired["levels"]], list(range(1, 9)))
            for policy, field in [("model", "wins"), ("random", "random_wins")]:
                count = sum(row[policy]["outcome"] == "通关" for row in paired["levels"])
                self.assertEqual(paired["summary"][field], count)
                self.assertEqual(paired["by_boxes"]["1"][field], count)

            args.compare_random = False
            run_evaluation(args, greedy, "cpu", 2, None)
            second, = set(reports.glob("*.json")) - {first}
            solo = json.loads(second.read_text(encoding="utf-8"))
            self.assertIsNone(solo["checkpoint_steps"])
            self.assertNotIn("random_wins", solo["summary"])
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
