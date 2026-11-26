"""Environment API for training and evaluation."""

from sokoban.engine import DIRECTIONS, Game, Point

from .board import ACTIONS, AREA, CHANNELS, SIDE, encode, legal_mask
from .deadlocks import is_deadlocked, pushed_into_deadlock
from .generation import make_training_game
from .rewards import DEFAULT_REVISIT_PENALTY, potential, shaped_reward, step_game

__all__ = [
    "ACTIONS", "AREA", "CHANNELS", "DEFAULT_REVISIT_PENALTY", "DIRECTIONS", "SIDE",
    "Game", "Point", "encode", "is_deadlocked", "legal_mask", "make_training_game",
    "potential", "pushed_into_deadlock", "shaped_reward", "step_game",
]
