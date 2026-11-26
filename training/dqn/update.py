"""Optimize the online network from one batch of replayed transitions."""

import random

import torch
from torch import nn
from torch.nn import functional as F

from ..profiling import null_region
from .batching import _tensor_batch, mask_batch, state_batch
from .replay import ReplayBuffer


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
