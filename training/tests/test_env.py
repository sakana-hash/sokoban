import random
import unittest
from unittest.mock import patch

from training.env import (
    ACTIONS, AREA, SIDE, Game, _pulls, encode, legal_mask, make_training_game, step_game,
)


class EnvironmentTests(unittest.TestCase):
    def test_encoding_preserves_overlays_and_rectangular_padding(self):
        game = Game(("########", "#+*$ . #", "#  $   #", "########"))
        data = encode(game)
        self.assertEqual(len(data), 4 * AREA)
        walls, goals, boxes, player = [data[i * AREA:(i + 1) * AREA] for i in range(4)]
        self.assertEqual(walls[SIDE + 1], 0)
        self.assertEqual(walls[SIDE + 8], 1)
        self.assertEqual(walls[-1], 1)
        self.assertEqual((goals[SIDE + 1], player[SIDE + 1]), (1, 1))
        self.assertEqual((goals[SIDE + 2], boxes[SIDE + 2]), (1, 1))
        self.assertEqual(sum(player), 1)
        self.assertEqual(sum(boxes), 3)
        self.assertEqual(sum(goals), 3)

    def test_rewards_masks_and_completion(self):
        game = Game(("######", "#@ $.#", "######"))
        self.assertEqual(legal_mask(game), (False, False, False, True))
        with self.assertRaises(ValueError):
            step_game(game, ACTIONS.index("up"))
        self.assertAlmostEqual(step_game(game, ACTIONS.index("right")), -0.01)
        self.assertAlmostEqual(step_game(game, ACTIONS.index("right")), 10.99)
        self.assertFalse(any(legal_mask(game)))
        game = Game(("########", "#@*  $.#", "########"))
        self.assertAlmostEqual(step_game(game, ACTIONS.index("right")), -1.01)

    def test_random_games_are_seeded_bounded_and_playable(self):
        first_rng, second_rng = random.Random(19), random.Random(19)
        layouts = set()
        with patch("sokoban.solver.solve", side_effect=AssertionError("solver called")):
            for limit in range(1, 5):
                for _ in range(30):
                    game = make_training_game(first_rng, limit)
                    self.assertEqual(encode(game), encode(make_training_game(second_rng, limit)))
                    self.assertFalse(game.won)
                    self.assertGreaterEqual(len(game.boxes), 1)
                    self.assertLessEqual(len(game.boxes), limit)
                    self.assertEqual(len(game.goals), len(game.boxes))
                    legal = [i for i, allowed in enumerate(legal_mask(game)) if allowed]
                    self.assertTrue(legal)
                    layouts.add(encode(game))
                    step_game(game, legal[0])
        self.assertGreater(len(layouts), 100)

    def test_each_reverse_pull_is_undone_by_a_single_push(self):
        box = (3, 3)
        reached = {(x, y) for x in range(1, 6) for y in range(1, 6)} - {box}
        options = _pulls({box}, reached)
        self.assertEqual(len(options), 4)
        for old, new, retreat in options:
            board = [list("#######")] + [list("#     #") for _ in range(5)] + [list("#######")]
            board[old[1]][old[0]] = "."
            board[new[1]][new[0]] = "$"
            board[retreat[1]][retreat[0]] = "@"
            game = Game(tuple("".join(row) for row in board))
            delta = (old[0] - new[0], old[1] - new[1])
            direction = {(0, -1): "up", (0, 1): "down", (-1, 0): "left", (1, 0): "right"}[delta]
            self.assertTrue(game.move(direction))
            self.assertTrue(game.won)


if __name__ == "__main__":
    unittest.main()
