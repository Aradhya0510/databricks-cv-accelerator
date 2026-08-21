"""Clean entry point for HF Trainer-based training.

Usage (single-node, native DDP — recommended):
    python jobs/train.py --config_path configs/detection_yolos_config.yaml
    python jobs/train.py --config_path configs/detection_yolos_config.yaml --num_gpus 4

Usage (multi-node via TorchDistributor — only when scaling across Spark workers):
    python jobs/train.py --config_path configs/detection_yolos_config.yaml --distributed torchd
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Support running this file straight from a repo checkout (Databricks Repos,
# ``python jobs/train.py``) as well as from an installed package.  Only the
# project root goes on the path — adding ``src/`` too would make both
# ``import config`` and ``import src.config`` resolve, to two different module
# objects.  Runtime dependencies are installed from main(), not at import time.
try:
    _this_file = Path(__file__).resolve()
except NameError:  # Databricks spark_python_task exec() context
    _this_file = Path(sys.argv[0]).resolve() if sys.argv else Path(os.getcwd())

_PROJECT_ROOT = _this_file.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    from src.utils.environment import ensure_runtime_requirements

    ensure_runtime_requirements(_PROJECT_ROOT / "requirements_runtime.txt")

    parser = argparse.ArgumentParser(description="Train a CV model with HF Trainer")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs (auto-detected if omitted)")
    parser.add_argument(
        "--distributed", type=str, default="native", choices=["native", "torchd"],
        help="'native' (default): HF Trainer handles DDP directly. "
             "'torchd': use TorchDistributor for multi-node Spark distribution.",
    )
    args = parser.parse_args()

    from src.config.schema import load_config
    from src.engine import TrainingEngine

    config = load_config(args.config_path)
    engine = TrainingEngine(config)
    metrics = engine.train(num_gpus=args.num_gpus, distributed_mode=args.distributed)

    print("\nTraining complete. Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
