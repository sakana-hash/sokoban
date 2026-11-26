"""Neural network, experience replay and a single Double DQN update."""

import random
from collections.abc import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .env import ACTIONS, SIDE
from .profiling import null_region


CHANNELS = 4
Mask = tuple[bool, ...]
Transition = tuple[bytes, int, float, bytes, Mask, bool]


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


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int, *, bottleneck: int | None = None):
        super().__init__()
        if bottleneck is None:
            first = nn.Conv2d(width, width, 3, dilation=dilation, padding=dilation)
        else:
            first = nn.Conv2d(width, bottleneck, 1)
        self.layers = nn.Sequential(
            first,
            nn.ReLU(),
            nn.Conv2d(first.out_channels, width, 3, dilation=dilation, padding=dilation),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.relu(features + self.layers(features))


class PlayerQNetwork(nn.Module):
    """Read four shared directional action values at the player's current square."""

    def __init__(self, *, extended: bool = False):
        super().__init__()
        width = 32
        layers = [
            nn.Conv2d(CHANNELS, width, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(width, dilation=1),
            ResidualBlock(width, dilation=3),
        ]
        if extended:
            layers.extend([
                ResidualBlock(width, dilation=1, bottleneck=4),
                ResidualBlock(width, dilation=1),
                ResidualBlock(width, dilation=3),
            ])
        self.features = nn.Sequential(*layers)
        self.head = nn.Conv2d(width, len(ACTIONS), 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        values = self.head(self.features(states))
        player = states[:, 3:4]
        return (values * player).sum(dim=(2, 3))


def make_network(version: int) -> nn.Module:
    if version == 1:
        return QNetwork()
    if version in (2, 3):
        return PlayerQNetwork(extended=version == 3)
    raise ValueError(f"Unknown network version {version!r}")


def state_batch(
    states: Iterable[bytes], device: torch.device, *, channels_last: bool = False, region=null_region,
) -> torch.Tensor:
    """Convert compact, channel-first boards to a batch of 0/1 floats."""
    with region("data_prep"):
        packed = bytearray().join(states)
        data = torch.frombuffer(packed, dtype=torch.uint8).reshape(-1, CHANNELS, SIDE, SIDE)
    with region("transfer"):
        layout = torch.channels_last if channels_last else torch.contiguous_format
        return data.to(device=device, dtype=torch.float32, memory_format=layout, non_blocking=True)


def _tensor_batch(items, dtype, device, region):
    with region("data_prep"):
        tensor = torch.tensor(items, dtype=dtype, device="cpu")
    with region("transfer"):
        return tensor.to(device, non_blocking=True)


def mask_batch(masks: Sequence[Mask], device: torch.device, *, region=null_region) -> torch.Tensor:
    """Batch legality of moving one square up, down, left or right."""
    return _tensor_batch(masks, torch.bool, device, region)


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


def optimize(
    online: nn.Module,
    target: nn.Module,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    rng: random.Random,
    *,
    channels_last: bool = False,
    region=null_region,
) -> torch.Tensor:
    with region("data_prep"):
        states, actions, rewards, next_states, next_masks, terminals = zip(*replay.sample(batch_size, rng))
    action_indices = _tensor_batch(actions, torch.long, device, region).unsqueeze(1)
    boards = state_batch(states, device, channels_last=channels_last, region=region)
    with region("train_compute"):
        predicted = online(boards).gather(1, action_indices).squeeze(1)
    with torch.no_grad():
        successors = state_batch(next_states, device, channels_last=channels_last, region=region)
        legal = mask_batch(next_masks, device, region=region)
        terminal = _tensor_batch(terminals, torch.bool, device, region)
        with region("train_compute"):
            # Online Q chooses the legal action. The delayed network values that action.
            next_actions = online(successors).masked_fill(~legal, -torch.inf).argmax(1, keepdim=True)
            next_values = target(successors).gather(1, next_actions).squeeze(1)
            bootstrap = next_values.masked_fill(terminal | ~legal.any(1), 0.0)
        immediate = _tensor_batch(rewards, torch.float32, device, region)
        with region("train_compute"):
            expected = immediate + gamma * bootstrap

    with region("train_compute"):
        loss = F.smooth_l1_loss(predicted, expected)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(online.parameters(), max_norm=10.0)
        optimizer.step()
    # Reporting reads this scalar at log/checkpoint time, without a GPU wait per update.
    return loss.detach()
