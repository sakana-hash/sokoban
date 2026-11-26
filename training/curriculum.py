"""Training start probabilities and validation for the box-count curriculum."""

from random import Random

from .env import SIDE, Game, make_training_game

SMALL_BOARD_MAX_SIDE = 7


def single_box_easy_probability(steps: int, *, anneal: bool = True) -> float:
    """Start at 75% easy; optionally fade to 25% between 50k and 200k moves."""
    if not anneal or steps <= 50_000:
        return 0.75
    progress = min(1.0, (steps - 50_000) / 150_000)
    return 0.75 - 0.50 * progress


def single_box_small_board_probability(steps: int) -> float:
    """Probability of restricting both dimensions to 5–7 for a new single-box room."""
    if steps <= 50_000:
        return 0.75
    return 0.75 * max(0, 150_000 - steps) / 100_000


def make_single_box_game(
    rng: Random, steps: int, *, small_board_curriculum: bool, anneal_easy_starts: bool = True,
) -> Game:
    easy_probability = single_box_easy_probability(steps, anneal=anneal_easy_starts)
    extra_pulls = 1 if rng.random() < easy_probability else None
    probability = single_box_small_board_probability(steps) if small_board_curriculum else 0.0
    max_side = SIDE
    if probability > 0 and rng.random() < probability:
        max_side = SMALL_BOARD_MAX_SIDE
    return make_training_game(rng, 1, max_extra_pulls=extra_pulls, max_side=max_side)


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
