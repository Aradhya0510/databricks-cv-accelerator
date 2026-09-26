"""Entry point for HF Trainer-based training on Databricks AI Runtime.

Usage (one process per GPU on this node):
    python jobs/train.py --config_path configs/detection_yolos_config.yaml

Usage (force a single process, e.g. for debugging on an 8-GPU node):
    python jobs/train.py --config_path configs/... --num_gpus 1

Submitted through ``databricks air run`` or an ``ai_runtime_task``, AI Runtime
runs this once per node.  When there is more than one process to start — more
than one GPU on the node, or more than one node — it stages the data, then
relaunches itself under ``torchrun``; each worker lands back here with
``LOCAL_RANK`` set and trains.  See ``src/engine/launch.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Support running this file straight from a checkout (a Git folder, the
# AI Runtime code snapshot at $CODE_SOURCE_PATH, ``python jobs/train.py``).
# Only the project root goes on the path — adding ``src/`` too would make both
# ``import config`` and ``import src.config`` resolve, to two different module
# objects.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a CV model with HF Trainer")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--num_gpus", type=int, default=None,
        help="Training processes per node (default: every GPU on the node)",
    )
    args = parser.parse_args()

    import yaml

    from src.config.schema import load_config
    from src.engine import TrainingEngine
    from src.engine.launch import run_torchrun
    from src.utils.environment import gpus_per_node, is_distributed_worker, is_rank_zero, num_nodes

    config = load_config(args.config_path)
    engine = TrainingEngine(config)

    if not is_distributed_worker():
        nproc = max(args.num_gpus or gpus_per_node(), 1)
        if nproc * num_nodes() > 1:
            # Stage once per node here, then hand the workers a config that
            # already points at the local copies.
            engine.stage_data()
            fd, resolved = tempfile.mkstemp(prefix="cv_config_", suffix=".yaml")
            with os.fdopen(fd, "w") as f:
                yaml.safe_dump(config.model_dump(mode="json"), f, sort_keys=False)
            return run_torchrun(__file__, ["--config_path", resolved], nproc)

    metrics = engine.train(num_gpus=1)

    if is_rank_zero():
        print("\nTraining complete. Metrics:")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
