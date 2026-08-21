"""Cross-rank reduction for custom evaluation loops.

The task eval closures iterate their own dataloader, which under DDP is a
shard of roughly ``1/world_size`` of the validation set.  Without reducing,
each rank reports a metric over its shard and rank 0's partial number drives
early stopping and best-checkpoint selection.

These helpers are no-ops when not running distributed, so the eval code reads
the same either way.
"""

from __future__ import annotations

from typing import Any, List

import torch


def is_distributed() -> bool:
    """True when a process group is initialised and has more than one rank."""
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """Sum *tensor* across ranks, returning a new tensor.

    Moves to the local CUDA device when NCCL is the backend, since NCCL
    cannot reduce CPU tensors.
    """
    if not is_distributed():
        return tensor

    backend = torch.distributed.get_backend()
    original_device = tensor.device

    work = tensor.clone()
    if backend == "nccl" and work.device.type != "cuda":
        work = work.cuda()

    torch.distributed.all_reduce(work, op=torch.distributed.ReduceOp.SUM)
    return work.to(original_device)


def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Concatenate *tensor* from every rank along dim 0.

    Ranks may hold different numbers of samples — the last shard is usually
    short — so lengths are exchanged first and the result is unpadded.
    """
    if not is_distributed():
        return tensor

    world = torch.distributed.get_world_size()
    backend = torch.distributed.get_backend()
    original_device = tensor.device

    work = tensor.cuda() if backend == "nccl" and tensor.device.type != "cuda" else tensor

    local_len = torch.tensor([work.shape[0]], device=work.device)
    lengths = [torch.zeros_like(local_len) for _ in range(world)]
    torch.distributed.all_gather(lengths, local_len)
    lengths = [int(l.item()) for l in lengths]

    max_len = max(lengths)
    if work.shape[0] < max_len:
        pad_shape = (max_len - work.shape[0],) + tuple(work.shape[1:])
        work = torch.cat([work, work.new_zeros(pad_shape)], dim=0)

    gathered = [torch.zeros_like(work) for _ in range(world)]
    torch.distributed.all_gather(gathered, work)

    trimmed = [g[:n] for g, n in zip(gathered, lengths)]
    return torch.cat(trimmed, dim=0).to(original_device)


def all_gather_objects(obj: Any) -> List[Any]:
    """Gather an arbitrary picklable object from every rank.

    Used for detection predictions and targets, which are lists of dicts of
    varying length that do not gather as a single tensor.
    """
    if not is_distributed():
        return [obj]

    world = torch.distributed.get_world_size()
    gathered: List[Any] = [None] * world
    torch.distributed.all_gather_object(gathered, obj)
    return gathered
