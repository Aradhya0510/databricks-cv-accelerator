"""Generate a monitoring report for a deployed endpoint.

Usage:
    python jobs/monitor.py --endpoint_name yolos-detection-endpoint
    python jobs/monitor.py --endpoint_name yolos-detection-endpoint --hours 48 --output_dir /tmp/reports
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
    parser = argparse.ArgumentParser(description="Monitor a deployed model endpoint")
    parser.add_argument("--endpoint_name", type=str, required=True, help="Serving endpoint name")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours")
    parser.add_argument("--output_dir", type=str, default="/tmp/monitoring", help="Report output directory")
    args = parser.parse_args()

    from src.monitoring import EndpointMonitor

    monitor = EndpointMonitor(args.endpoint_name)

    # 1. Health check
    print("\n" + "=" * 60)
    print("ENDPOINT HEALTH")
    print("=" * 60)
    health = monitor.get_health()
    print(f"  Endpoint: {health['endpoint_name']}")
    print(f"  Ready:    {health['ready']}")
    for m in health.get("served_models", []):
        print(f"  Model:    {m['entity_name']} v{m['entity_version']} ({m['workload_size']})")

    # 2. Request metrics
    print("\n" + "=" * 60)
    print(f"REQUEST METRICS (last {args.hours}h)")
    print("=" * 60)
    req_metrics = monitor.get_request_metrics(hours=args.hours)
    if "error" not in req_metrics:
        print(f"  Total requests: {req_metrics.get('total_requests', 0)}")
        print(f"  Error rate:     {req_metrics.get('error_rate', 0):.2%}")
        print(f"  Avg latency:    {req_metrics.get('avg_latency_ms', 0):.0f} ms")
        print(f"  P95 latency:    {req_metrics.get('p95_latency_ms', 0):.0f} ms")
    else:
        print(f"  Error querying metrics: {req_metrics['error']}")

    # 3. Prediction distribution
    print("\n" + "=" * 60)
    print(f"PREDICTION DISTRIBUTION (last {args.hours}h)")
    print("=" * 60)
    pred_dist = monitor.get_prediction_distribution(hours=args.hours)
    if "error" not in pred_dist:
        print(f"  Responses sampled: {pred_dist.get('num_responses_sampled', 0)}")
        conf = pred_dist.get("confidence_stats", {})
        print(f"  Avg confidence:    {conf.get('mean', 0):.3f}")
        class_dist = pred_dist.get("class_distribution", {})
        if class_dist:
            top_classes = sorted(class_dist.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"  Top classes: {top_classes}")
    else:
        print(f"  Error: {pred_dist['error']}")

    # 4. Full report
    os.makedirs(args.output_dir, exist_ok=True)
    report_path = os.path.join(args.output_dir, f"monitoring_report_{args.endpoint_name}.json")
    monitor.generate_report(output_path=report_path)

    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
