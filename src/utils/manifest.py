"""The hand-off from training to the evaluate and deploy steps.

AI Runtime tasks cannot pass job task values, so the training run's identity
is written to a small JSON file in ``output.results_dir``.  When that
directory is a UC Volume, a downstream task in the same job — or a later
manual run — can find the model without being told a run ID.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

MANIFEST_NAME = "run_manifest.json"


def manifest_path(results_dir: str) -> str:
    return os.path.join(results_dir, MANIFEST_NAME)


def write_run_manifest(results_dir: str, **fields: Any) -> str:
    """Atomically write *fields* as the run manifest and return its path."""
    os.makedirs(results_dir, exist_ok=True)
    path = manifest_path(results_dir)
    partial = f"{path}.partial"
    with open(partial, "w") as f:
        json.dump(fields, f, indent=2, sort_keys=True)
    os.replace(partial, path)
    return path


def read_run_manifest(results_dir: str) -> Dict[str, Any]:
    """Read the manifest written by the last training run into *results_dir*."""
    path = manifest_path(results_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No run manifest at {path}. Train first, or pass --run_id / "
            f"--model_uri explicitly. For a multi-task job, output.results_dir "
            f"must be a UC Volume path so every task sees the same file."
        )
    with open(path) as f:
        return json.load(f)
