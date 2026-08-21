"""The contract between training and registration for model artifacts.

Training must persist enough for registration and serving to rebuild the model
*and its preprocessing*.  Getting only half of that across was the original
break: training logged a bare model (which ``mlflow.transformers`` rejects
outright), and registration then tried to read an image processor that had
never been saved.

Two layouts are supported, and the layout used is recorded as a run tag so the
loader never has to guess:

``transformers_flavor``
    ``mlflow.transformers.log_model`` with the component-dict form.  Preferred:
    it produces a real MLflow 3 LoggedModel with a ``models:/`` URI, and stores
    the model under ``model/`` and the processor under
    ``components/image_processor/``.

``flat_directory``
    ``model.save_pretrained()`` + ``processor.save_pretrained()`` into one
    directory, logged with ``mlflow.log_artifacts``.  The fallback for
    architectures the transformers flavor has no pipeline task for.

Either way :func:`resolve_model_dir` hands back a single local directory that
``AutoModelForX.from_pretrained`` and ``AutoImageProcessor.from_pretrained``
both accept — which is exactly what registration and the PyFunc wrappers want.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Optional, Tuple

import mlflow

# Run param carrying the logged model URI (MLflow 3 LoggedModel).
LOGGED_MODEL_PARAM = "logged_model_uri"

# Run tag recording which of the two layouts above was written.
ARTIFACT_LAYOUT_TAG = "model_artifact_layout"

FLAVOR_LAYOUT = "transformers_flavor"
FLAT_LAYOUT = "flat_directory"

# Maps our task names onto the transformers pipeline task the flavor needs.
_PIPELINE_TASKS = {
    "detection": "object-detection",
    "classification": "image-classification",
    "segmentation": "image-segmentation",
}


def log_model_artifacts(
    model: Any,
    processor: Any,
    *,
    task_type: str,
    artifact_name: str = "model",
) -> str:
    """Log *model* and *processor* together and return the artifact URI.

    Tries the transformers flavor first and falls back to a flat directory,
    tagging the run either way so :func:`resolve_model_dir` knows what it is
    reading.  The processor is never optional — a model logged without its
    preprocessing cannot be served correctly.
    """
    if processor is None:
        raise ValueError(
            "processor is required: a model logged without its image processor "
            "cannot be reloaded for registration or served with the same "
            "preprocessing it was trained with."
        )

    pipeline_task = _PIPELINE_TASKS.get(task_type)

    if pipeline_task is not None:
        try:
            model_info = _log_transformers_flavor(
                model, processor, pipeline_task, artifact_name
            )
            mlflow.set_tag(ARTIFACT_LAYOUT_TAG, FLAVOR_LAYOUT)
            return model_info.model_uri
        except Exception as exc:  # noqa: BLE001 - fall back, but say why
            print(
                f"transformers flavor unavailable for task '{pipeline_task}' "
                f"({type(exc).__name__}: {exc}); "
                f"falling back to a flat directory artifact."
            )

    uri = _log_flat_directory(model, processor, artifact_name)
    mlflow.set_tag(ARTIFACT_LAYOUT_TAG, FLAT_LAYOUT)
    return uri


def _log_transformers_flavor(model, processor, pipeline_task: str, artifact_name: str):
    """Log via ``mlflow.transformers``, handling the MLflow 2 / 3 kwarg rename."""
    import inspect

    payload = {"model": model, "image_processor": processor}
    kwargs = {"transformers_model": payload, "task": pipeline_task}

    # MLflow 3 renamed ``artifact_path`` to ``name``.
    if "name" in inspect.signature(mlflow.transformers.log_model).parameters:
        kwargs["name"] = artifact_name
    else:
        kwargs["artifact_path"] = artifact_name

    return mlflow.transformers.log_model(**kwargs)


def _log_flat_directory(model, processor, artifact_name: str) -> str:
    """Save model + processor side by side and log the directory."""
    tmpdir = tempfile.mkdtemp(prefix="cv_model_")
    model.save_pretrained(tmpdir)
    processor.save_pretrained(tmpdir)
    mlflow.log_artifacts(tmpdir, artifact_path=artifact_name)

    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("log_model_artifacts requires an active MLflow run")
    return f"runs:/{run.info.run_id}/{artifact_name}"


def resolve_model_dir(
    run_id: Optional[str] = None,
    model_uri: Optional[str] = None,
    artifact_name: str = "model",
) -> str:
    """Return a local directory holding both the model and its processor.

    Resolution order is *model_uri*, then the URI recorded on *run_id*, then
    the ``runs:/`` fallback.  Whichever layout was written, the returned
    directory is flat and ready for ``from_pretrained``.
    """
    uri, layout = _resolve_uri_and_layout(run_id, model_uri, artifact_name)

    local_path = mlflow.artifacts.download_artifacts(artifact_uri=uri)

    if _looks_flat(local_path):
        return local_path

    # Transformers-flavor layout: let MLflow reassemble the components rather
    # than hardcoding its internal directory names, then flatten them.
    if layout != FLAT_LAYOUT:
        try:
            return _flatten_flavor_components(uri)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not resolve model artifacts from '{uri}'. The download "
                f"at {local_path} is not a flat HuggingFace directory and "
                f"loading it as transformers components failed: {exc}"
            ) from exc

    raise RuntimeError(
        f"Artifact at '{uri}' has no config.json and is not a transformers "
        f"flavor model. Contents: {sorted(os.listdir(local_path))}"
    )


def _resolve_uri_and_layout(
    run_id: Optional[str],
    model_uri: Optional[str],
    artifact_name: str,
) -> Tuple[str, Optional[str]]:
    if model_uri:
        return model_uri, None

    if not run_id:
        raise ValueError("resolve_model_dir needs either run_id or model_uri")

    client = mlflow.MlflowClient()
    run = client.get_run(run_id)

    layout = run.data.tags.get(ARTIFACT_LAYOUT_TAG)
    stored_uri = run.data.params.get(LOGGED_MODEL_PARAM)
    if stored_uri:
        return stored_uri, layout

    return f"runs:/{run_id}/{artifact_name}", layout


def _looks_flat(path: str) -> bool:
    """True when *path* is directly loadable by ``from_pretrained``."""
    return os.path.isfile(os.path.join(path, "config.json"))


def _flatten_flavor_components(uri: str) -> str:
    """Load a transformers-flavor model and re-save it as a flat directory."""
    components = mlflow.transformers.load_model(uri, return_type="components")

    model = components.get("model")
    if model is None:
        raise RuntimeError(f"No 'model' component in {sorted(components)}")

    processor = (
        components.get("image_processor")
        or components.get("feature_extractor")
        or components.get("processor")
    )

    out = tempfile.mkdtemp(prefix="cv_model_flat_")
    model.save_pretrained(out)
    if processor is not None:
        processor.save_pretrained(out)
    return out
