"""Uniform replay storage and its checkpoint format."""

import random


Mask = tuple[bool, ...]
Transition = tuple[bytes, int, float, bytes, Mask, bool]


class ReplayBuffer:
    """Uniform replay over the last capacity transitions; storage grows on demand."""

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self._items: list[Transition] = []
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(
        self,
        state: bytes,
        action: int,
        reward: float,
        next_state: bytes,
        next_mask: Mask,
        terminal: bool,
    ) -> None:
        transition = (state, action, reward, next_state, next_mask, terminal)
        if len(self) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._cursor] = transition
        self._cursor = (self._cursor + 1) % self.capacity

    def sample(self, count: int, rng: random.Random) -> list[Transition]:
        return rng.sample(self._items, count)

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "cursor": self._cursor, "items": self._items.copy()}

    def load_state_dict(self, state: dict) -> None:
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity differs from checkpoint")
        items = state["items"]
        cursor = state["cursor"]
        if len(items) > self.capacity or not 0 <= cursor < self.capacity:
            raise ValueError("Invalid replay checkpoint")
        if len(items) < self.capacity and cursor != len(items):
            raise ValueError("Invalid replay checkpoint cursor")
        self._items = list(items)
        self._cursor = cursor
