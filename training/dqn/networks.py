"""Q networks for the three supported checkpoint versions."""

import torch
from torch import nn
from torch.nn import functional as F

from ..env import ACTIONS, CHANNELS, SIDE


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
