"""Cross-rank reduction, exercised against a real process group.

These spawn a two-rank gloo group on CPU, so the reduction is tested for
real rather than mocked. Under DDP each rank evaluates a shard of the
validation set; without reducing, the reported metric covers only part of
the data and rank 0's partial number drives early stopping.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402


def _run(rank: int, world_size: int, fn, queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29517"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        result = fn(rank, world_size)
        if rank == 0:
            queue.put(result)
    finally:
        dist.destroy_process_group()


def _spawn(fn, world_size: int = 2):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.start_processes(
        _run, args=(world_size, fn, queue), nprocs=world_size,
        join=True, start_method="spawn",
    )
    return queue.get(timeout=30)


# ---------------------------------------------------------------------------
# Worker functions must be module-level to be picklable by spawn.
# ---------------------------------------------------------------------------

def _sum_worker(rank, world_size):
    from src.utils.distributed import all_reduce_sum

    # rank 0 contributes [1, 2], rank 1 contributes [10, 20]
    local = torch.tensor([1, 2], dtype=torch.long) * (10 ** rank)
    return all_reduce_sum(local).tolist()


def _gather_worker(rank, world_size):
    from src.utils.distributed import all_gather_tensor

    # Uneven shards: rank 0 has 3 items, rank 1 has 2.
    local = torch.tensor([0, 1, 2]) if rank == 0 else torch.tensor([10, 11])
    return sorted(all_gather_tensor(local).tolist())


def _gather_objects_worker(rank, world_size):
    from src.utils.distributed import all_gather_objects

    local = [{"rank": rank, "n": i} for i in range(rank + 1)]
    gathered = all_gather_objects(local)
    return [item for chunk in gathered for item in chunk]


def _segmentation_worker(rank, world_size):
    from src.tasks.segmentation.metrics import SegmentationMetrics

    m = SegmentationMetrics(num_classes=2)
    if rank == 0:
        # 4 pixels, all correct
        m.update(torch.tensor([0, 0, 1, 1]), torch.tensor([0, 0, 1, 1]))
    else:
        # 4 pixels, all wrong
        m.update(torch.tensor([1, 1, 0, 0]), torch.tensor([0, 0, 1, 1]))
    return m.compute()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_op_without_a_process_group():
    """Eval code reads the same whether or not it is running distributed."""
    from src.utils.distributed import all_gather_tensor, all_reduce_sum, is_distributed

    assert is_distributed() is False
    t = torch.tensor([1, 2, 3])
    assert torch.equal(all_reduce_sum(t), t)
    assert torch.equal(all_gather_tensor(t), t)


@pytest.mark.slow
def test_all_reduce_sums_across_ranks():
    assert _spawn(_sum_worker) == [11, 22]


@pytest.mark.slow
def test_all_gather_handles_uneven_shards():
    """The last shard is usually short, so lengths must be exchanged."""
    assert _spawn(_gather_worker) == [0, 1, 2, 10, 11]


@pytest.mark.slow
def test_all_gather_objects_collects_every_rank():
    result = _spawn(_gather_objects_worker)
    # rank 0 contributes 1 item, rank 1 contributes 2.
    assert len(result) == 3
    assert {item["rank"] for item in result} == {0, 1}


@pytest.mark.slow
def test_segmentation_metrics_reduce_over_the_whole_set():
    """Rank 0 alone would report 100% accuracy; the full set is 50%."""
    metrics = _spawn(_segmentation_worker)
    assert metrics["eval_pixel_accuracy"] == pytest.approx(0.5)
