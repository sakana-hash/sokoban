"""Training start probabilities and validation for the box-count curriculum."""

from random import Random

from .env import make_training_game


def single_box_easy_probability(steps: int) -> float:
    """Hold the opening mix for 50k moves, then fade it to 25% easy by 200k."""
    if steps <= 50_000:
        return 0.75
    progress = min(1.0, (steps - 50_000) / 150_000)
    return 0.75 - 0.50 * progress


def validation_games(per_box_count: int):
    for count in range(1, 5):
        rng = Random(971_000 + count)
        accepted = 0
        while accepted < per_box_count:
            game = make_training_game(rng, count)
            if len(game.boxes) == count:
                accepted += 1
                yield count, game


def next_stage(stage: int, wins: list[int], per_box_count: int) -> int:
    if stage < 4 and all(4 * n >= 3 * per_box_count for n in wins[:stage]):
        return stage + 1
    return stage
