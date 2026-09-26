"""Process, GPU and data-locality helpers for training on Databricks AI Runtime."""

from __future__ import annotations

import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# ---------------------------------------------------------------------------
# Process topology
# ---------------------------------------------------------------------------

def is_distributed_worker() -> bool:
    """True inside a process started by a distributed launcher.

    ``torchrun`` and ``serverless_gpu``'s ``@distributed`` both set
    ``LOCAL_RANK`` per process.  ``WORLD_SIZE`` is not a usable signal: AI
    Runtime sets it on every node of a multi-node task *before* any launcher
    runs, so a node-level script would mistake itself for a worker.
    """
    return "LOCAL_RANK" in os.environ


def is_rank_zero() -> bool:
    """True when this process is global rank 0 (or not running distributed).

    Uses ``RANK`` rather than ``LOCAL_RANK`` so that in multi-node runs exactly
    one process across the whole job — not one per node — is treated as the
    writer of MLflow runs, checkpoints and reports.
    """
    return int(os.environ.get("RANK", "0")) == 0


def num_nodes() -> int:
    """Nodes in this AI Runtime task; AI Runtime sets ``NUM_NODES`` on multi-node runs."""
    return int(os.environ.get("NUM_NODES", "1"))


def get_gpu_count() -> int:
    """Return number of NVIDIA GPUs via nvidia-smi (avoids importing torch/CUDA init)."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            return len(lines)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def gpus_per_node() -> int:
    """GPUs this node should run one process each on.

    Prefers ``LOCAL_WORLD_SIZE``, which AI Runtime sets from the accelerator
    type on multi-node tasks, and falls back to counting devices.
    """
    local = os.environ.get("LOCAL_WORLD_SIZE")
    if local:
        return int(local)
    return get_gpu_count()


# ---------------------------------------------------------------------------
# Data staging: /Volumes → local disk
# ---------------------------------------------------------------------------
#
# UC Volumes are readable from every process on AI Runtime, but they are tuned
# for large sequential reads.  Image datasets are the opposite — tens of
# thousands of small files, re-read every epoch — so they are copied once to
# local disk with a parallel copy, which is what the AI Runtime data-loading
# guidance recommends for small-file workloads.

_VOLUMES_ROOT = "/Volumes"

# Written last into a staged directory; its absence means the copy is partial.
_STAGED_MARKER = ".cv_staging_complete"

_COPY_CONCURRENCY = 64


def volumes_staging_path(volumes_path: str, local_root: str) -> str:
    """Map a ``/Volumes/...`` path to its deterministic local staging path.

    Split out from :func:`stage_data_to_local` so the path arithmetic can be
    tested without touching the filesystem.

    NOTE: this must use ``removeprefix``, not ``lstrip``.  ``lstrip`` strips
    *characters* — the set ``{/, V, o, l, u, m, e, s}`` — so it keeps eating
    into the first real path segment: ``/Volumes/main/cv`` became ``ain/cv``
    and ``/Volumes/mount/x`` became ``nt/x``, which could collide two distinct
    sources onto one staging directory.
    """
    relative = volumes_path.removeprefix("/Volumes/").rstrip("/")
    return os.path.join(local_root, relative)


def stage_data_to_local(volumes_path: str, local_root: str = "/tmp/cv_data") -> str:
    """Copy a ``/Volumes/`` file or directory tree to local disk, once.

    Paths outside ``/Volumes/`` are returned unchanged.  A directory counts as
    staged only once its completion marker exists, so a copy interrupted by a
    preempted or timed-out run is redone rather than trained on half-empty.

    Must be called from one process per node — the launcher, before any
    workers exist — never concurrently from every rank.

    Returns:
        The local path that should replace the original volumes path.
    """
    if not volumes_path.startswith("/Volumes/"):
        return volumes_path

    local_path = volumes_staging_path(volumes_path, local_root)
    # ``_VOLUMES_ROOT`` is indirected so tests can rebase the source tree.
    src = Path(_VOLUMES_ROOT) / volumes_path.removeprefix("/Volumes/").rstrip("/")

    if src.is_file():
        if not os.path.isfile(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            partial = f"{local_path}.partial"
            shutil.copyfile(src, partial)
            os.replace(partial, local_path)
            print(f"Staged {volumes_path} → {local_path}")
        return local_path

    if os.path.isfile(os.path.join(local_path, _STAGED_MARKER)):
        return local_path

    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    _parallel_copy_tree(src, Path(local_path))
    Path(local_path, _STAGED_MARKER).touch()
    print(f"Staged {volumes_path} → {local_path}")
    return local_path


def _parallel_copy_tree(src: Path, dest: Path) -> None:
    files = [p for p in src.rglob("*") if p.is_file()]
    for directory in {dest / f.parent.relative_to(src) for f in files} | {dest}:
        directory.mkdir(parents=True, exist_ok=True)

    def _copy(f: Path) -> None:
        shutil.copyfile(f, dest / f.relative_to(src))

    with ThreadPoolExecutor(max_workers=_COPY_CONCURRENCY) as pool:
        # list() re-raises the first copy error instead of dropping it.
        list(pool.map(_copy, files))


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------

def resolve_precision(requested: str = "auto") -> str:
    """Resolve a precision setting against the hardware actually present.

    Every AI Runtime accelerator (A10, H100, B300) supports bf16, but the
    check stays so CPU runs and local development resolve sensibly.

    Returns one of ``"bf16"``, ``"fp16"`` or ``"fp32"``.
    """
    import torch

    if not torch.cuda.is_available():
        return "fp32"

    def _bf16_ok() -> bool:
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            return False

    if requested == "auto":
        return "bf16" if _bf16_ok() else "fp16"

    if requested == "bf16" and not _bf16_ok():
        print(
            "Warning: precision 'bf16' requested but this GPU does not support "
            "it; falling back to fp16."
        )
        return "fp16"

    return requested
