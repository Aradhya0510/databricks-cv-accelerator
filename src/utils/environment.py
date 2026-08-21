"""Environment detection and GPU helpers for Databricks training."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def is_databricks_job() -> bool:
    """True when running inside a Databricks *Jobs* environment (non-interactive)."""
    return os.getenv("DATABRICKS_JOB_RUN_ID") is not None


def is_databricks_notebook() -> bool:
    """True when running inside a Databricks *notebook* (interactive)."""
    return (
        os.getenv("DATABRICKS_RUNTIME_VERSION") is not None
        and not is_databricks_job()
    )


def is_databricks() -> bool:
    return os.getenv("DATABRICKS_RUNTIME_VERSION") is not None


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# NCCL / distributed setup
# ---------------------------------------------------------------------------

def setup_nccl_env() -> None:
    """Set NCCL environment variables suitable for single-node Databricks DDP."""
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
    os.environ.setdefault("NCCL_SHM_DISABLE", "1")


# ---------------------------------------------------------------------------
# Data staging for /Volumes/ → /tmp/ (DDP workers can't access FUSE)
# ---------------------------------------------------------------------------

_VOLUMES_ROOT = "/Volumes"


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


def stage_data_to_local(
    volumes_path: str,
    local_root: str = "/tmp/staged_data",
) -> str:
    """Copy a /Volumes/ directory tree to a local path for DDP worker access.

    If *volumes_path* does not start with ``/Volumes/`` the path is returned
    unchanged (nothing to stage).

    Returns:
        The local path that should replace the original volumes path.
    """
    if not volumes_path.startswith("/Volumes/"):
        return volumes_path

    local_path = volumes_staging_path(volumes_path, local_root)

    if os.path.exists(local_path):
        return local_path

    # ``_VOLUMES_ROOT`` is indirected so tests can rebase the source tree.
    src = Path(_VOLUMES_ROOT) / volumes_path.removeprefix("/Volumes/").rstrip("/")

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if src.is_file():
        shutil.copy2(str(src), local_path)
    else:
        shutil.copytree(str(src), local_path)

    print(f"Staged {volumes_path} → {local_path}")
    return local_path


# ---------------------------------------------------------------------------
# Runtime dependency installation
# ---------------------------------------------------------------------------

def is_rank_zero() -> bool:
    """True when this process is global rank 0 (or not running distributed).

    Uses ``RANK`` rather than ``LOCAL_RANK`` so that in multi-node runs exactly
    one process across the whole job — not one per node — is treated as the
    writer of MLflow runs, checkpoints and reports.
    """
    return int(os.environ.get("RANK", "0")) == 0


def ensure_runtime_requirements(requirements_file: str | Path) -> None:
    """Install ``requirements_file`` if it exists, once, on rank 0 only.

    Databricks job clusters should normally declare these as cluster libraries;
    this is the fallback for running the entry points directly from a checkout.
    Set ``CV_SKIP_RUNTIME_INSTALL=1`` to opt out entirely.

    Called explicitly from ``main()`` in the job entry points — never at import
    time, so importing a job module has no side effects.
    """
    if os.environ.get("CV_SKIP_RUNTIME_INSTALL"):
        return
    if not is_rank_zero():
        return

    path = Path(requirements_file)
    if not path.exists():
        return

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(path)],
        stdout=subprocess.DEVNULL,
    )


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------

def resolve_precision(requested: str = "auto") -> str:
    """Resolve a precision setting against the hardware actually present.

    ``bf16`` needs Ampere or newer; V100 and T4 are still common on Databricks
    GPU pools and used to fail outright at TrainingArguments construction
    because bf16 was hardcoded to ``torch.cuda.is_available()``.

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
