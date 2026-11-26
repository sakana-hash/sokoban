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

    def test_walk_revisits_cost_extra_until_a_box_moves(self):
        game = Game(("#######", "#@ $ .#", "#     #", "#######"))
        visited = {game.player}
        moves = (
            ("down", -0.01), ("right", -0.01), ("up", -0.01),
            ("left", -0.03),  # A four-step loop back to the starting square.
            ("right", -0.03), ("left", -0.03), ("right", -0.03),
            ("right", -0.01),  # Push the box one square without reaching the goal.
            ("left", -0.01),  # Previously visited before the push.
            ("right", -0.03),  # Return to the player's square just after the push.
            ("right", 10.99),
        )
        for step, (direction, reward) in enumerate(moves, 1):
            with self.subTest(step=step):
                self.assertAlmostEqual(step_game(game, ACTIONS.index(direction), visited), reward)

    def test_random_games_are_bounded_and_playable(self):
        rng = random.Random(19)
        with patch("sokoban.solver.solve", side_effect=AssertionError("solver called")):
            for limit in range(1, 5):
                game = make_training_game(rng, limit)
                self.assertFalse(game.won)
                self.assertIn(len(game.boxes), range(1, limit + 1))
                step_game(game, legal_mask(game).index(True))

    def test_configured_revisit_penalty(self):
        for penalty, expected in ((0.04, -0.05), (0.0, -0.01)):
            with self.subTest(penalty=penalty):
                game = Game(("#######", "#@  $.#", "#######"))
                visited = {game.player}
                outward = step_game(game, ACTIONS.index("right"), visited, revisit_penalty=penalty)
                inward = step_game(game, ACTIONS.index("left"), visited, revisit_penalty=penalty)
                self.assertAlmostEqual(outward, -0.01)
                self.assertAlmostEqual(inward, expected)


if __name__ == "__main__":
    unittest.main()
