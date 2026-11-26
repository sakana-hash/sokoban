"""Prepare compact board observations and legal-action masks for PyTorch."""

from collections.abc import Iterable, Sequence

import torch

from ..env import CHANNELS, SIDE
from ..profiling import null_region
from .replay import Mask


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
