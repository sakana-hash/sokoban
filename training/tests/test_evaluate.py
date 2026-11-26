import unittest

from training.env import ACTIONS, Game
from training.evaluate import evaluation_boards, play_batch, play_level
from training.runtime import checkpoint_version


class EvaluationTests(unittest.TestCase):
    def test_old_and_new_checkpoint_versions(self):
        for version in (1, 2):
            self.assertEqual(checkpoint_version({"version": version, "online": {}}), version)
        with self.assertRaises(ValueError):
            checkpoint_version({"version": 9, "online": {}})

    def test_room_draws_are_reproducible_and_allow_independent_replay(self):
        rooms = list(evaluation_boards(10, 1, 73))
        self.assertEqual(rooms, list(evaluation_boards(10, 1, 73)))
        self.assertNotEqual(rooms, list(evaluation_boards(10, 1, 74)))
        self.assertEqual(len(rooms), 10)
        for _, rows in rooms:
            game, baseline = Game(rows), Game(rows)
            self.assertEqual(len(game.boxes), 1)
            game.move(game.observation()["legal_actions"][0])
            self.assertEqual(baseline.observation()["board"], list(rows))
        self.assertEqual(len(list(evaluation_boards(None, 1, 73))), 30)

    def test_stochastic_revisits_can_still_win(self):
        rows = ("#######", "# @ $.#", "#######")

        def follow_directions():
            sequence = iter(["left", "right", "right", "right"])
            return lambda state, mask: ACTIONS.index(next(sequence))

        self.assertEqual(play_level(Game(rows), follow_directions(), 4), "循环")
        self.assertEqual(
            play_level(Game(rows), follow_directions(), 4, detect_cycle=False), "通关"
        )

    def test_batch_shrinks_and_matches_individual_play(self):
        boards = [
            ("######", "#@$. #", "######"),
            ("#######", "#@$ . #", "#######"),
            ("######", "#@ * #", "######"),
        ]
        games = [Game(board) for board in boards]
        sizes = []

        def choose_right(state, mask):
            return max(i for i, allowed in enumerate(mask) if allowed)

        def choose_batch(states, masks):
            sizes.append(len(states))
            return [choose_right(s, m) for s, m in zip(states, masks, strict=True)]

        outcomes = play_batch(games, choose_batch, 3)
        self.assertEqual(sizes, [2, 1])
        self.assertEqual(outcomes, ["通关"] * 3)
        for board, game, outcome in zip(boards, games, outcomes, strict=True):
            individual = Game(board)
            self.assertEqual(play_level(individual, choose_right, 3), outcome)
            self.assertEqual(individual.observation(), game.observation())

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
