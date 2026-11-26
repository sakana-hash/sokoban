import unittest

from training.env import ACTIONS, Game
from training.evaluate import evaluation_boards, play_level


class EvaluationTests(unittest.TestCase):
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
