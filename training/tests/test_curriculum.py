import unittest
from collections import Counter
from random import Random

from training.curriculum import (
    make_single_box_game, next_stage, single_box_easy_probability,
    single_box_small_board_probability, validation_games,
)
from training.env import SIDE, encode, make_training_game


class CurriculumTests(unittest.TestCase):
    def test_single_box_start_probabilities(self):
        for steps, easy, small in (
            (0, 0.75, 0.75), (50_000, 0.75, 0.75), (125_000, 0.5, 0.1875),
            (150_000, 5 / 12, 0.0), (200_000, 0.25, 0.0), (1_000_000, 0.25, 0.0),
        ):
            with self.subTest(steps=steps):
                self.assertAlmostEqual(single_box_easy_probability(steps), easy)
                self.assertEqual(single_box_easy_probability(steps, anneal=False), 0.75)
                self.assertAlmostEqual(single_box_small_board_probability(steps), small)

    def test_small_board_sampling_is_independent_of_easy_starts(self):
        training_rng, reference_rng = Random(42), Random(42)
        combinations = set()
        for _ in range(100):
            easy = reference_rng.random() < 0.75
            small = reference_rng.random() < 0.75
            combinations.add((easy, small))
            max_side = 7 if small else SIDE
            expected = make_training_game(
                reference_rng, 1, max_extra_pulls=1 if easy else None, max_side=max_side,
            )
            game = make_single_box_game(training_rng, 0, small_board_curriculum=True)
            board = game.observation()["board"]
            self.assertEqual(board, expected.observation()["board"])
            self.assertIn(len(board), range(5, max_side + 1))
            self.assertIn(len(board[0]), range(5, max_side + 1))
        self.assertEqual(len(combinations), 4)

    def test_regular_board_sampling_preserves_seeded_starts(self):
        for steps, enabled, anneal in ((0, False, True), (150_000, True, True), (400_000, False, False)):
            with self.subTest(steps=steps, small_board_curriculum=enabled, anneal=anneal):
                training_rng, reference_rng = Random(42), Random(42)
                for _ in range(5):
                    probability = single_box_easy_probability(steps) if anneal else 0.75
                    easy = reference_rng.random() < probability
                    expected = make_training_game(reference_rng, 1, max_extra_pulls=1 if easy else None)
                    game = make_single_box_game(
                        training_rng, steps, small_board_curriculum=enabled, anneal_easy_starts=anneal,
                    )
                    self.assertEqual(encode(game), encode(expected))

    def test_validation_is_repeatable_and_balanced(self):
        first = [(n, encode(game)) for n, game in validation_games(5)]
        second = [(len(game.boxes), encode(game)) for _, game in validation_games(5)]
        self.assertEqual(first, second)
        self.assertEqual(Counter(n for n, _ in first), {1: 5, 2: 5, 3: 5, 4: 5})

    def test_promotion_requires_each_unlocked_group(self):
        self.assertEqual(next_stage(1, [23, 32, 32, 32], 32), 1)
        self.assertEqual(next_stage(1, [24, 0, 0, 0], 32), 2)
        self.assertEqual(next_stage(2, [22, 32, 32, 32], 32), 2)
        self.assertEqual(next_stage(3, [24, 24, 24, 0], 32), 4)
        self.assertEqual(next_stage(4, [32, 32, 32, 32], 32), 4)
