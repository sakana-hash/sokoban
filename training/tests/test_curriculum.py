import unittest
from collections import Counter

from training.curriculum import next_stage, validation_games
from training.env import encode


class CurriculumTests(unittest.TestCase):
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
