"""Saved hyperparameters for a Double DQN training run."""

from dataclasses import dataclass

from .env import DEFAULT_REVISIT_PENALTY


@dataclass(frozen=True)
class Config:
    seed: int = 42
    num_envs: int = 16
    batch_size: int = 64
    replay_size: int = 50_000
    warmup: int = 2_000
    epsilon_steps: int = 150_000
    curriculum_steps: int = 10_000
    max_episode_steps: int = 256
    validation_rooms: int = 32
    revisit_penalty: float = DEFAULT_REVISIT_PENALTY
    deadlock_penalty: float = -1.0
    small_board_curriculum: bool = True
    model_version: int = 2
    deadlock_detection: bool = True
    anneal_easy_starts: bool = True
