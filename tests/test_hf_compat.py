"""The transformers v5 contract.

Two v5 ``from_pretrained`` defaults would change training and eval numerics
without changing any calling code, so ``src.utils.hf`` pins both.  These tests
hold that pinning in place: a regression here is silent everywhere else, showing
up only as metrics that no longer match a published baseline.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.utils.hf import (  # noqa: E402
    IMAGE_PROCESSOR_BACKEND,
    MODEL_DTYPE,
    load_image_processor,
    require_transformers_v5,
)


# ---------------------------------------------------------------------------
# Version guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("version", ["4.51.3", "4.44.0", "3.0.2"])
def test_pre_v5_is_rejected_with_an_actionable_message(version):
    """The DBR ML runtime ships 4.x, so this is the misconfiguration to catch."""
    with pytest.raises(RuntimeError, match="requirements_runtime.txt"):
        require_transformers_v5(version)


@pytest.mark.parametrize("version", ["5.0.0", "5.0.0rc1", "5.2.1.dev0", "6.0.0"])
def test_v5_and_newer_are_accepted(version):
    require_transformers_v5(version)


def test_an_unparseable_version_is_left_alone():
    """A source build reports something like 'main'; don't block the developer."""
    require_transformers_v5("main")


def test_the_installed_version_satisfies_the_guard():
    """Importing src.utils.hf already ran this, so it is really an assertion
    that the test environment matches what the framework requires."""
    require_transformers_v5()


# ---------------------------------------------------------------------------
# Pinned defaults
# ---------------------------------------------------------------------------

def test_weights_load_in_float32():
    """v5 defaults to the checkpoint's saved precision. Mixed precision is HF
    Trainer's job via bf16/fp16; the master weights must stay float32."""
    assert MODEL_DTYPE is torch.float32


def test_the_processor_backend_is_pinned_to_torchvision():
    """v5 picks a backend from what happens to be installed, and the torchvision
    and PIL backends resize differently."""
    assert IMAGE_PROCESSOR_BACKEND == "torchvision"


def test_load_image_processor_passes_the_pinned_backend(monkeypatch):
    captured = {}

    class FakeAutoImageProcessor:
        @staticmethod
        def from_pretrained(source, **kwargs):
            captured["source"] = source
            captured["kwargs"] = kwargs
            return "processor"

    import transformers

    monkeypatch.setattr(
        transformers, "AutoImageProcessor", FakeAutoImageProcessor, raising=True,
    )

    result = load_image_processor("some/model", size={"height": 8, "width": 8})

    assert result == "processor"
    assert captured["source"] == "some/model"
    assert captured["kwargs"]["backend"] == IMAGE_PROCESSOR_BACKEND
    assert captured["kwargs"]["size"] == {"height": 8, "width": 8}


def test_an_explicit_backend_overrides_the_default(monkeypatch):
    """A family whose torchvision processor lacks a needed capability can opt out."""
    captured = {}

    class FakeAutoImageProcessor:
        @staticmethod
        def from_pretrained(source, **kwargs):
            captured.update(kwargs)
            return "processor"

    import transformers

    monkeypatch.setattr(
        transformers, "AutoImageProcessor", FakeAutoImageProcessor, raising=True,
    )

    load_image_processor("some/model", backend="pil")

    assert captured["backend"] == "pil"


# ---------------------------------------------------------------------------
# Removed architectures
# ---------------------------------------------------------------------------

def test_deta_is_no_longer_a_known_detection_family():
    """DETA was deleted in v5 along with the rest of models/deprecated, so the
    adapter must not advertise support for it."""
    from src.tasks.detection.adapters import _FAMILY_CONFIGS, detect_detection_family

    assert "deta" not in _FAMILY_CONFIGS

    # It must also not be captured by a neighbouring pattern such as "detr".
    family, _ = detect_detection_family("jozhang97/deta-swin-large")
    assert family == "generic"
