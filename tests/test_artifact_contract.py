"""The training → registration → serving artifact contract.

This is the seam the pipeline used to break at: training logged a bare model
(rejected by ``mlflow.transformers``) and registration then read an image
processor that had never been saved.  These tests round-trip a real — if tiny —
model through the whole chain on CPU, with no network.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile

import pytest

mlflow = pytest.importorskip("mlflow")
torch = pytest.importorskip("torch")

from src.serving.artifacts import (  # noqa: E402
    ARTIFACT_LAYOUT_TAG,
    LOGGED_MODEL_PARAM,
    _FLAVOR_PIP_REQUIREMENTS,
    _log_transformers_flavor,
    log_model_artifacts,
    resolve_model_dir,
)


@pytest.fixture(scope="module", autouse=True)
def tracking_uri():
    """A throwaway sqlite tracking store; the file store is deprecated."""
    d = tempfile.mkdtemp(prefix="mlflow_test_")
    mlflow.set_tracking_uri(f"sqlite:///{d}/mlflow.db")
    mlflow.set_experiment("artifact-contract-tests")
    yield


@pytest.fixture
def tiny_classifier():
    """A 2-class ViT built in code — small enough to train in CI, no download."""
    from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

    cfg = ViTConfig(
        image_size=32, patch_size=16, num_hidden_layers=1, num_attention_heads=2,
        hidden_size=32, intermediate_size=64, num_labels=2,
    )
    model = ViTForImageClassification(cfg)
    processor = ViTImageProcessor(size={"height": 32, "width": 32})
    return model, processor


def test_logging_without_a_processor_is_rejected(tiny_classifier):
    """A model logged alone cannot be served correctly, so refuse it up front."""
    model, _ = tiny_classifier
    with mlflow.start_run():
        with pytest.raises(ValueError, match="processor is required"):
            log_model_artifacts(model, None, task_type="classification")


def test_round_trip_yields_a_flat_loadable_directory(tiny_classifier):
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    model, processor = tiny_classifier

    with mlflow.start_run() as run:
        uri = log_model_artifacts(model, processor, task_type="classification")
        mlflow.log_param(LOGGED_MODEL_PARAM, uri)
        run_id = run.info.run_id

    resolved = resolve_model_dir(run_id=run_id)

    # Both halves must be loadable straight from the resolved directory —
    # this is exactly what registration and the PyFunc wrapper do.
    reloaded = AutoModelForImageClassification.from_pretrained(resolved)
    reloaded_processor = AutoImageProcessor.from_pretrained(resolved)

    assert reloaded.config.num_labels == 2
    assert reloaded_processor is not None


def test_resolve_works_from_the_uri_alone(tiny_classifier):
    from transformers import AutoImageProcessor

    model, processor = tiny_classifier
    with mlflow.start_run():
        uri = log_model_artifacts(model, processor, task_type="classification")

    resolved = resolve_model_dir(model_uri=uri)
    assert AutoImageProcessor.from_pretrained(resolved) is not None


def test_layout_is_recorded_on_the_run(tiny_classifier):
    model, processor = tiny_classifier
    with mlflow.start_run() as run:
        log_model_artifacts(model, processor, task_type="classification")
        run_id = run.info.run_id

    tags = mlflow.MlflowClient().get_run(run_id).data.tags
    assert tags[ARTIFACT_LAYOUT_TAG] in {"transformers_flavor", "flat_directory"}


def test_label_names_survive_the_round_trip():
    """id2label is what lets serving return 'cat' instead of 'LABEL_0'."""
    from transformers import (
        AutoModelForImageClassification, ViTConfig, ViTForImageClassification,
        ViTImageProcessor,
    )
    from src.utils.labels import apply_label_names

    model = ViTForImageClassification(ViTConfig(
        image_size=32, patch_size=16, num_hidden_layers=1, num_attention_heads=2,
        hidden_size=32, intermediate_size=64, num_labels=2,
    ))
    apply_label_names(model, ["cat", "dog"], 2)
    processor = ViTImageProcessor(size={"height": 32, "width": 32})

    with mlflow.start_run():
        uri = log_model_artifacts(model, processor, task_type="classification")

    resolved = resolve_model_dir(model_uri=uri)
    reloaded = AutoModelForImageClassification.from_pretrained(resolved)
    assert reloaded.config.id2label[0] == "cat"
    assert reloaded.config.id2label[1] == "dog"


def test_the_flavor_does_not_ask_serving_for_tensorflow():
    """MLflow infers a transformers model's requirements by probing for framework
    base classes.  Its probe for ``FlaxPreTrainedModel`` raises under v5 (Flax is
    gone), and MLflow then hedges by requiring both PyTorch *and* TensorFlow.  We
    declare the requirements instead, so assert the declaration stays sane."""
    joined = " ".join(_FLAVOR_PIP_REQUIREMENTS).lower()
    assert "tensorflow" not in joined
    assert "transformers>=5" in joined
    # The pinned processor backend is torchvision, so serving needs it present.
    assert "torchvision" in joined


def test_the_flavor_requirements_are_actually_forwarded(monkeypatch, tiny_classifier):
    """A declaration that never reaches log_model would silently re-enable
    MLflow's inference, so pin the wiring, not just the list."""
    model, processor = tiny_classifier
    captured = {}

    def fake_log_model(
        *, transformers_model=None, task=None, name=None, pip_requirements=None,
    ):
        captured.update(
            transformers_model=transformers_model, task=task, name=name,
            pip_requirements=pip_requirements,
        )
        return type("Info", (), {"model_uri": "models:/fake"})()

    monkeypatch.setattr(mlflow.transformers, "log_model", fake_log_model)

    _log_transformers_flavor(model, processor, "image-classification", "model")

    assert captured["pip_requirements"] == _FLAVOR_PIP_REQUIREMENTS
    assert captured["name"] == "model"


def test_registration_rejects_an_artifact_with_no_processor(tmp_path):
    """The guard that turns a silent preprocessing mismatch into a clear error."""
    from transformers import ViTConfig, ViTForImageClassification
    from src.serving.registration import _assert_processor_present

    model_dir = tmp_path / "model_only"
    ViTForImageClassification(ViTConfig(
        image_size=32, patch_size=16, num_hidden_layers=1, num_attention_heads=2,
        hidden_size=32, intermediate_size=64, num_labels=2,
    )).save_pretrained(model_dir)

    with pytest.raises(RuntimeError, match="No image processor config"):
        _assert_processor_present(str(model_dir))


def test_classification_pyfunc_serves_the_resolved_artifact(tiny_classifier):
    """End to end: log -> resolve -> PyFunc predict on a real base64 image."""
    from PIL import Image
    from src.serving.pyfunc import ClassificationPyFuncModel
    from src.utils.labels import apply_label_names

    model, processor = tiny_classifier
    apply_label_names(model, ["cat", "dog"], 2)

    with mlflow.start_run():
        uri = log_model_artifacts(model, processor, task_type="classification")
    model_dir = resolve_model_dir(model_uri=uri)

    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(10, 200, 10)).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    wrapper = ClassificationPyFuncModel()
    wrapper.load_context(type("Ctx", (), {"artifacts": {"model_dir": model_dir}})())
    result = wrapper.predict(None, [{"image": b64}])

    assert result[0]["predictions"]["status"] == "success"
    assert result[0]["predictions"]["label_name"] in {"cat", "dog"}
