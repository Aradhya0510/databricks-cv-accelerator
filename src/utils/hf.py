"""Shared HuggingFace loading helpers.

Transformers v5 relaxed two ``from_pretrained`` defaults in ways that change
numerics without changing any calling code, so both are pinned here rather than
inherited from the library:

``dtype``
    v5 loads a checkpoint in whatever precision it was *saved* in
    (``dtype="auto"``), where v4 forced float32.  Fine-tuning needs float32
    master weights: mixed precision is applied by HF Trainer through the
    ``bf16``/``fp16`` TrainingArguments, and letting a bf16-saved checkpoint
    become the master copy degrades the optimizer state rather than speeding
    anything up.  Serving pins it too — endpoints are often CPU-only, where a
    half-precision checkpoint is slow at best and unimplemented at worst.

``backend``
    v5 replaced the slow/fast image-processor split with named backends and
    defaults to torchvision whenever torchvision is installed, which on DBR ML
    it always is.  The torchvision and PIL backends resize differently, so
    leaving the choice implicit means a library upgrade can move eval metrics
    with no change to this repo.  Pinning it makes preprocessing reproducible
    and keeps training and serving on the same path.
"""

from __future__ import annotations

from typing import Any

import torch

_REQUIRED_MAJOR = 5


def require_transformers_v5(version: str | None = None) -> None:
    """Fail fast when the environment predates the v5 APIs used here.

    Databricks Runtime ML still ships transformers 4.x, so a cluster that never
    installed ``requirements_runtime.txt`` would otherwise fail much later and
    much less legibly — a ``TypeError`` about an unexpected ``backend`` keyword
    deep inside ``from_pretrained``.

    Args:
        version: The version to check. Reads the installed one when omitted;
            passing it explicitly is for tests.
    """
    if version is None:
        import transformers

        version = transformers.__version__

    try:
        major = int(version.split(".", 1)[0])
    except ValueError:
        # Unparseable version means a source build; assume it is intentional.
        return

    if major < _REQUIRED_MAJOR:
        raise RuntimeError(
            f"This framework requires transformers >= {_REQUIRED_MAJOR}, but "
            f"found {version}. Databricks Runtime ML still ships 4.x, so the "
            f"pinned runtime dependencies must be installed on the cluster: "
            f"`pip install -r requirements_runtime.txt`, or declare them as "
            f"cluster libraries in the job definition."
        )


# Checked on import so that it trips before any model or processor is built —
# every task routes its processor construction through this module.
require_transformers_v5()


# The precision to load pretrained weights in. See the module docstring.
MODEL_DTYPE = torch.float32

# The image-processor implementation to preprocess with. See the module docstring.
IMAGE_PROCESSOR_BACKEND = "torchvision"


def load_image_processor(source: str, *, backend: str | None = None, **kwargs: Any):
    """``AutoImageProcessor.from_pretrained`` with the backend pinned.

    Args:
        source: A hub model id or a local directory.
        backend: Overrides :data:`IMAGE_PROCESSOR_BACKEND`.  Needed only by a
            model family whose torchvision processor lacks a capability the
            task depends on.
        **kwargs: Forwarded to ``from_pretrained``.
    """
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(
        source,
        backend=backend or IMAGE_PROCESSOR_BACKEND,
        **kwargs,
    )
