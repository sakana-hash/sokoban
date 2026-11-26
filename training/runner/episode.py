"""State, rewards and snapshots for one self-play episode."""

from collections.abc import Iterable

from ..env import (
    Game, Point, encode, legal_mask, potential, pushed_into_deadlock, shaped_reward, step_game,
)


DISCOUNT = 0.99


class Episode:
    def __init__(
        self, game: Game, version: int, length: int = 0, visited_since_push: Iterable[Point] = (),
    ):
        self.game = game
        self.version = version
        self.length = length
        self.state = encode(game)
        self.mask = legal_mask(game)
        self.potential = potential(game) if version >= 2 else 0.0
        self.visited_since_push = {game.player, *visited_since_push} if version >= 2 else None

    def snapshot(self) -> dict:
        snapshot = {"board": self.game.observation()["board"], "length": self.length}
        if self.visited_since_push is not None:
            snapshot["visited_since_push"] = sorted(self.visited_since_push)
        return snapshot

    def step(self, action: int, config) -> tuple[float, bool, bool]:
        """Apply one action and return its reward, terminal flag and deadlock flag."""
        previous_boxes = self.game.boxes
        reward = step_game(
            self.game, action, self.visited_since_push, revisit_penalty=config.revisit_penalty,
        )
        self.length += 1
        self.state = encode(self.game)
        self.mask = legal_mask(self.game)
        won = self.game.won
        deadlocked = config.deadlock_detection and not won and pushed_into_deadlock(self.game, previous_boxes)
        if deadlocked:
            reward += config.deadlock_penalty
        terminal = won or deadlocked or not any(self.mask)
        if self.version >= 2:
            after = potential(self.game)
            reward = shaped_reward(reward, self.potential, after, terminal, DISCOUNT)
            self.potential = after
        return reward, terminal, deadlocked
