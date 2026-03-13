"""Environment detection and GPU helpers for Databricks training."""

from __future__ import annotations

import os
import shutil
import subprocess
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

    # Deterministic local name based on the volumes path
    relative = volumes_path.lstrip("/Volumes/").rstrip("/")
    local_path = os.path.join(local_root, relative)

    if os.path.exists(local_path):
        return local_path

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    src = Path(volumes_path)
    if src.is_file():
        shutil.copy2(str(src), local_path)
    else:
        shutil.copytree(str(src), local_path)

    print(f"Staged {volumes_path} → {local_path}")
    return local_path
