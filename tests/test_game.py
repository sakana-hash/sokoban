import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sokoban import LEVELS, Game, load_level, solve
from sokoban.server import play_request


class GameTests(unittest.TestCase):
    def test_campaign_solutions_replay_in_the_game(self):
        self.assertEqual(len(LEVELS), 30)
        self.assertEqual(len({tuple(rows) for _, rows in LEVELS}), 30)
        for level in range(1, 31):
            with self.subTest(level=level):
                game = load_level(level)
                start = game.observation()
                self.assertFalse(game.won)
                solution = solve(game)
                self.assertIsNotNone(solution)
                self.assertEqual(game.observation(), start)
                for action in solution:
                    self.assertTrue(game.move(action))
                self.assertTrue(game.won)
                self.assertEqual(game.moves, len(solution))

    def test_blocked_moves_and_undo_across_a_goal(self):
        game = Game(("######", "#@ $.#", "######"))
        initial = game.observation()
        self.assertFalse(game.move("left"))
        self.assertFalse(game.undo())
        self.assertTrue(game.move("right"))
        before_push = game.observation()
        self.assertTrue(game.move("right"))
        self.assertEqual(game.observation()["board"][1], "#  @*#")
        self.assertTrue(game.won)
        self.assertTrue(game.undo())
        self.assertEqual(game.observation(), before_push)
        game.reset()
        self.assertEqual(game.observation(), initial)
        self.assertFalse(game.undo())

    def test_two_adjacent_boxes_cannot_be_pushed_together(self):
        game = Game(("########", "#@$$ ..#", "########"))
        self.assertFalse(game.move("right"))
        self.assertEqual(game.moves, 0)
        self.assertIsNone(solve(game))

    def test_solver_uses_current_position_in_browser_api(self):
        prefix = ["right"]
        request = {"level": 1, "actions": prefix}
        state = play_request({**request, "command": "solve"})
        self.assertFalse(state["won"])
        final = play_request({**request, "actions": prefix + state["solution"]})
        self.assertTrue(final["won"])
        self.assertEqual(final["moves"], 2)


if __name__ == "__main__":
    unittest.main()
