import random
import unittest
from unittest.mock import patch

from training.env import (
    ACTIONS, AREA, SIDE, Game, _pulls, encode, legal_mask, make_training_game, potential,
    shaped_reward, step_game,
)


class EnvironmentTests(unittest.TestCase):
    def test_potential_rewards_walking_and_primitive_pushes(self):
        from sokoban import load_level

        gamma = 0.99
        with patch("sokoban.solver.solve", side_effect=AssertionError("solver called")):
            game = load_level(1)
            initial = potential(game)
            base = step_game(game, ACTIONS.index("right"))
            after_push = potential(game)
            self.assertEqual((game.moves, game.pushes), (1, 1))
            self.assertGreater(shaped_reward(base, initial, after_push, game.won, gamma), base)
            base = step_game(game, ACTIONS.index("right"))
            self.assertTrue(game.won)
            self.assertEqual(potential(game), 0.0)
            self.assertAlmostEqual(shaped_reward(base, after_push, 0, True, gamma), base - after_push)

            game = Game(("########", "#@  $ .#", "########"))
            before = potential(game)
            step_game(game, ACTIONS.index("right"))
            self.assertEqual((game.moves, game.pushes), (1, 0))
            self.assertGreater(potential(game), before)

    def test_discounted_shaping_telescopes_on_a_closed_walk(self):
        game = Game(("########", "#@  $ .#", "########"))
        gamma = 0.99
        start = potential(game)
        step_game(game, ACTIONS.index("right"))
        middle = potential(game)
        step_game(game, ACTIONS.index("left"))
        end = potential(game)
        self.assertEqual(end, start)
        outward = shaped_reward(0, start, middle, False, gamma)
        inward = shaped_reward(0, middle, end, False, gamma)
        self.assertAlmostEqual(outward + gamma * inward, gamma ** 2 * end - start)
        self.assertEqual(shaped_reward(0, middle, -1, True, gamma), -middle)

    def test_potential_is_bounded_with_no_eligible_standing_square(self):
        game = Game(("#######", "#$ @ .#", "#######"))
        self.assertAlmostEqual(potential(game), -0.1 * 4 - 0.04 * 18)

    def test_initial_lesson_uses_exactly_one_reverse_displacement(self):
        rng = random.Random(321)
        for _ in range(50):
            game = make_training_game(rng, 1, max_extra_pulls=0)
            (x, y), = game.boxes
            (gx, gy), = game.goals
            self.assertEqual(abs(x - gx) + abs(y - gy), 1)
        for limit in range(1, 5):
            original = make_training_game(random.Random(7), limit)
            explicit = make_training_game(random.Random(7), limit, max_extra_pulls=4 * limit)
            self.assertEqual(encode(original), encode(explicit))

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
