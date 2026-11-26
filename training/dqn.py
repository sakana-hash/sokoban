"""Neural network, experience replay and a single Double DQN update."""

import random
from collections.abc import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from .env import ACTIONS, SIDE


CHANNELS = 4
Transition = tuple[bytes, int, float, bytes, tuple[bool, ...], bool]


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * SIDE * SIDE, 128),
            nn.ReLU(),
            nn.Linear(128, len(ACTIONS)),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)


def state_batch(states: Iterable[bytes], device: torch.device) -> torch.Tensor:
    """Convert compact, channel-first boards to a batch of 0/1 floats."""
    packed = bytearray().join(states)
    data = torch.frombuffer(packed, dtype=torch.uint8)
    return data.reshape(-1, CHANNELS, SIDE, SIDE).to(device=device, dtype=torch.float32)


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
        next_mask: tuple[bool, ...],
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


def optimize(
    online: nn.Module,
    target: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    rng: random.Random,
) -> float:
    states, actions, rewards, next_states, next_masks, terminals = zip(
        *replay.sample(batch_size, rng)
    )
    action_indices = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
    predicted = online(state_batch(states, device)).gather(1, action_indices).squeeze(1)

    with torch.no_grad():
        successors = state_batch(next_states, device)
        legal = torch.tensor(next_masks, dtype=torch.bool, device=device)
        terminal = torch.tensor(terminals, dtype=torch.bool, device=device)
        # Online Q chooses the legal action. The delayed network values that action.
        next_actions = online(successors).masked_fill(~legal, -torch.inf).argmax(1, keepdim=True)
        next_values = target(successors).gather(1, next_actions).squeeze(1)
        bootstrap = next_values.masked_fill(terminal | ~legal.any(1), 0.0)
        expected = torch.tensor(rewards, dtype=torch.float32, device=device) + gamma * bootstrap

    loss = F.smooth_l1_loss(predicted, expected)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
    optimizer.step()
    return loss.item()
