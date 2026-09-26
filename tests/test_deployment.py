"""Endpoint deployment against the installed Databricks SDK.

The request objects are built with the real SDK dataclasses, so a signature
change in a newer SDK fails here rather than halfway through a deploy job.
"""

from __future__ import annotations

import pytest

pytest.importorskip("databricks.sdk")

from src.serving.deployment import deploy_endpoint  # noqa: E402


class _FakeEndpoints:
    def __init__(self, exists: bool = False):
        self.exists = exists
        self.calls = []

    def create(self, name, config):
        self.calls.append(("create", name, config))
        if self.exists:
            raise RuntimeError("RESOURCE_ALREADY_EXISTS: endpoint exists")

    def update_config(self, name, served_entities):
        self.calls.append(("update_config", name, served_entities))


@pytest.fixture
def endpoints(monkeypatch):
    fake = _FakeEndpoints()
    client = type("Client", (), {"serving_endpoints": fake})
    monkeypatch.setattr("databricks.sdk.WorkspaceClient", lambda: client)
    return fake


def test_create_builds_a_valid_config(endpoints):
    result = deploy_endpoint("cv-endpoint", "main.cv.model", model_version="3")

    assert result == {"endpoint_name": "cv-endpoint", "status": "created"}
    (_, name, config), = endpoints.calls
    assert name == "cv-endpoint"
    (entity,) = config.served_entities
    assert (entity.entity_name, entity.entity_version) == ("main.cv.model", "3")
    assert entity.scale_to_zero_enabled is True


def test_an_existing_endpoint_is_updated(endpoints):
    endpoints.exists = True

    result = deploy_endpoint("cv-endpoint", "main.cv.model", model_version="4")

    assert result["status"] == "updated"
    assert endpoints.calls[-1][0] == "update_config"
    assert endpoints.calls[-1][2][0].entity_version == "4"
