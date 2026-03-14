"""Evaluate a trained CV model: mAP metrics, error analysis, benchmarks.

Usage:
    python jobs/evaluate.py --config_path configs/detection_yolos_config.yaml --checkpoint_path /path/to/model
    python jobs/evaluate.py --config_path configs/detection_yolos_config.yaml --run_id abc123
    python jobs/evaluate.py --config_path configs/detection_yolos_config.yaml --checkpoint_path /path/to/model --max_batches 50
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    _this_file = Path(__file__).resolve()
except NameError:
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
    parser = argparse.ArgumentParser(description="Evaluate a trained CV model")
    parser.add_argument("--config_path", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--run_id", type=str, default=None, help="MLflow run ID to load model from")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Local model checkpoint path")
    parser.add_argument("--max_batches", type=int, default=None, help="Max batches for evaluation")
    parser.add_argument("--output_dir", type=str, default=None, help="Override results output directory")
    args = parser.parse_args()

    from src.config.schema import load_config
    from src.evaluation import EvaluationEngine

    config = load_config(args.config_path)
    if args.output_dir:
        config.output.results_dir = args.output_dir

    engine = EvaluationEngine(config)

    # 1. Metrics
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    metrics = engine.evaluate(
        model_path=args.checkpoint_path,
        run_id=args.run_id,
        max_batches=args.max_batches,
    )
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 2. Error analysis
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    errors = engine.error_analysis(
        model_path=args.checkpoint_path,
        run_id=args.run_id,
        max_batches=args.max_batches or 100,
    )
    for k, v in errors["summary"].items():
        print(f"  {k}: {v}")

    # 3. Benchmark
    print("\n" + "=" * 60)
    print("BENCHMARK")
    print("=" * 60)
    bench = engine.benchmark(
        model_path=args.checkpoint_path,
        run_id=args.run_id,
    )
    print(f"  FPS:              {bench['fps']:.1f}")
    print(f"  Latency p50 (ms): {bench['latency_per_batch_ms']['p50']:.1f}")
    print(f"  Latency p95 (ms): {bench['latency_per_batch_ms']['p95']:.1f}")
    print(f"  Latency p99 (ms): {bench['latency_per_batch_ms']['p99']:.1f}")
    if "gpu_memory_mb" in bench:
        print(f"  GPU Memory (MB):  {bench['gpu_memory_mb']:.0f}")

    print(f"\nResults saved to: {config.output.results_dir}")


if __name__ == "__main__":
    main()
