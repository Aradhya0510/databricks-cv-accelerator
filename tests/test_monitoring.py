"""Monitoring queries and threshold evaluation."""

from __future__ import annotations

import inspect
import re

from src.monitoring.endpoint_monitor import EndpointMonitor


class _Thresholds:
    error_rate_threshold = 0.05
    latency_p95_threshold_ms = 500
    drift_threshold = 0.1


def _monitor() -> EndpointMonitor:
    """An EndpointMonitor without a workspace client (no Databricks needed)."""
    m = EndpointMonitor.__new__(EndpointMonitor)
    m.endpoint_name = "test-endpoint"
    m.thresholds = _Thresholds()
    return m


def test_no_sql_is_built_by_string_interpolation():
    """Endpoint names reach these queries from CLI args and app form fields."""
    source = inspect.getsource(EndpointMonitor)
    for method in ("get_request_metrics", "get_prediction_distribution"):
        body = source.split(f"def {method}")[1].split("\n    def ")[0]
        assert 'f"""' not in body, f"{method} builds SQL with an f-string"
        assert "{self.endpoint_name}" not in body, f"{method} interpolates the endpoint name"
        assert ":endpoint_name" in body, f"{method} should bind :endpoint_name"


def test_error_rate_breach_is_reported():
    breaches = _monitor().evaluate_thresholds({"error_rate": 0.2, "p95_latency_ms": 100})
    assert [b["metric"] for b in breaches] == ["error_rate"]
    assert breaches[0]["value"] == 0.2
    assert breaches[0]["threshold"] == 0.05


def test_latency_breach_is_reported():
    breaches = _monitor().evaluate_thresholds({"error_rate": 0.0, "p95_latency_ms": 900})
    assert [b["metric"] for b in breaches] == ["p95_latency_ms"]


def test_healthy_metrics_produce_no_breaches():
    assert _monitor().evaluate_thresholds({"error_rate": 0.01, "p95_latency_ms": 120}) == []


def test_both_thresholds_can_breach_at_once():
    breaches = _monitor().evaluate_thresholds({"error_rate": 0.9, "p95_latency_ms": 9000})
    assert len(breaches) == 2


def test_no_thresholds_configured_means_no_breaches():
    m = _monitor()
    m.thresholds = None
    assert m.evaluate_thresholds({"error_rate": 1.0, "p95_latency_ms": 99999}) == []


def test_missing_metrics_are_skipped_not_treated_as_zero():
    """A failed query returns {"error": ...} — that must not read as healthy."""
    assert _monitor().evaluate_thresholds({"error": "warehouse unavailable"}) == []
