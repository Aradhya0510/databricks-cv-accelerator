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

# Support running this file straight from a checkout (a Git folder, a bundle
# deployment, ``python jobs/monitor.py``).  Only the project root goes on the
# path — adding ``src/`` too would make both ``import config`` and
# ``import src.config`` resolve, to two different module objects.
try:
    _this_file = Path(__file__).resolve()
except NameError:  # Databricks spark_python_task exec() context
    _this_file = Path(sys.argv[0]).resolve() if sys.argv else Path(os.getcwd())

_PROJECT_ROOT = _this_file.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Monitor a deployed model endpoint")
    parser.add_argument("--endpoint_name", type=str, required=True, help="Serving endpoint name")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours")
    parser.add_argument("--output_dir", type=str, default="/tmp/monitoring", help="Report output directory")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Pipeline config supplying the monitoring thresholds")
    parser.add_argument("--fail_on_breach", action="store_true",
                        help="Exit non-zero when a threshold is breached, so a "
                             "scheduled job alerts instead of silently passing")
    args = parser.parse_args()

    from src.monitoring import EndpointMonitor

    thresholds = None
    if args.config_path:
        from src.config.schema import load_config

        thresholds = load_config(args.config_path).monitoring

    monitor = EndpointMonitor(args.endpoint_name, thresholds=thresholds)

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
    report = monitor.generate_report(output_path=report_path)

    print(f"\nFull report saved to: {report_path}")

    breaches = report.get("threshold_breaches", [])
    if breaches:
        print("\n" + "=" * 60)
        print("THRESHOLD BREACHES")
        print("=" * 60)
        for b in breaches:
            print(f"  {b['metric']}: {b['value']} exceeds {b['threshold']}")
        if args.fail_on_breach:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
