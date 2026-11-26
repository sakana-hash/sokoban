import random
import unittest
from unittest.mock import patch

from training.env import (
    ACTIONS, AREA, SIDE, Game, encode, legal_mask, make_training_game, potential,
    shaped_reward, step_game,
)


class EnvironmentTests(unittest.TestCase):
    def test_discounted_shaping_telescopes_on_a_closed_walk(self):
        game = Game(("########", "#@  $ .#", "########"))
        gamma = 0.99
        start = potential(game)
        step_game(game, ACTIONS.index("right"))
        middle = potential(game)
        step_game(game, ACTIONS.index("left"))
        end = potential(game)
        self.assertGreater(middle, start)
        self.assertEqual(end, start)
        outward = shaped_reward(0, start, middle, False, gamma)
        inward = shaped_reward(0, middle, end, False, gamma)
        self.assertAlmostEqual(outward + gamma * inward, gamma ** 2 * end - start)
        self.assertEqual(shaped_reward(0, middle, -1, True, gamma), -middle)

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

    def test_random_games_are_bounded_and_playable(self):
        rng = random.Random(19)
        with patch("sokoban.solver.solve", side_effect=AssertionError("solver called")):
            for limit in range(1, 5):
                game = make_training_game(rng, limit)
                self.assertFalse(game.won)
                self.assertIn(len(game.boxes), range(1, limit + 1))
                step_game(game, legal_mask(game).index(True))


if __name__ == "__main__":
    unittest.main()
