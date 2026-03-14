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

# Ensure src/ is importable.
# On Databricks, spark_python_task runs via exec() where __file__ is not defined.
# Fall back to inspecting the script path from sys.argv or the config_path parent.
try:
    _this_file = Path(__file__).resolve()
except NameError:
    # Databricks exec() context — derive project root from the python_file path
    # that Spark passes as the first positional element before argparse runs.
    _this_file = Path(sys.argv[0]).resolve() if sys.argv else Path(os.getcwd())

_project_root = _this_file.parent.parent
sys.path.insert(0, str(_project_root / "src"))
sys.path.insert(0, str(_project_root))

_runtime_reqs = _project_root / "requirements_runtime.txt"
if _runtime_reqs.exists():
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(_runtime_reqs)],
        stdout=subprocess.DEVNULL,
    )


def main():
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
