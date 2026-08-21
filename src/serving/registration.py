"""Model registration: save artifacts, log PyFunc, validate, register to UC."""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.models import infer_signature

from .artifacts import resolve_model_dir


def _set_uc_registry() -> None:
    """Point the MLflow registry at Unity Catalog.

    Three-level ``catalog.schema.model`` names are only valid against the UC
    registry.  Relying on the workspace default meant registration failed with
    an opaque name-format error anywhere that default was not already UC.
    """
    try:
        if mlflow.get_registry_uri() != "databricks-uc":
            mlflow.set_registry_uri("databricks-uc")
    except Exception as exc:  # noqa: BLE001 - non-Databricks tracking backends
        print(f"Note: could not set the Unity Catalog registry URI ({exc}).")


_PROCESSOR_FILES = (
    "preprocessor_config.json",
    "image_processor_config.json",
    "processor_config.json",
)


def _assert_processor_present(model_dir: str) -> None:
    """Fail loudly when the artifact has no preprocessing config.

    Serving an image model without the processor it was trained with silently
    changes resize and normalisation, so this is worth catching at
    registration rather than discovering from degraded predictions.
    """
    if any(os.path.isfile(os.path.join(model_dir, f)) for f in _PROCESSOR_FILES):
        return
    raise RuntimeError(
        f"No image processor config found in {model_dir}. The training run did "
        f"not log the processor alongside the model, so the served model would "
        f"preprocess differently than it was trained. Contents: "
        f"{sorted(os.listdir(model_dir))}"
    )


def register_model(
    run_id: str,
    registered_model_name: str,
    *,
    task_type: str = "detection",
    model_uri: Optional[str] = None,
    aliases: Optional[List[str]] = None,
    tags: Optional[Dict[str, str]] = None,
    validate: bool = True,
    test_image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Log a PyFunc model to MLflow and register it to Unity Catalog.

    Args:
        run_id: MLflow run ID whose model artifact to wrap.
        registered_model_name: Three-level UC name (catalog.schema.model).
        task_type: "detection" or "classification" — selects PyFunc wrapper.
        model_uri: Direct model URI from log_model (preferred in MLflow 3).
                   When omitted, resolved automatically from the run.
        aliases: Aliases to set on the new version (e.g. ["champion", "latest"]).
        tags: Tags to attach to the model version.
        validate: If True, run a local prediction test before registering.
        test_image_path: Optional path to a real image for validation.

    Returns:
        Dict with model_uri, model_version, and registered_model_name.
    """
    aliases = aliases or ["champion", "latest"]
    tags = tags or {}

    _set_uc_registry()

    # 1. Resolve a flat local directory holding both the model and its
    #    processor.  The artifact contract guarantees both are present, so
    #    there is no need to reload and re-save them here.
    artifact_path = resolve_model_dir(run_id=run_id, model_uri=model_uri)
    _assert_processor_present(artifact_path)

    # 2. Copy into a clean directory that becomes the PyFunc's artifact.
    tmpdir = tempfile.mkdtemp(prefix="cv_pyfunc_")
    model_dir = os.path.join(tmpdir, "model_artifacts")
    shutil.copytree(artifact_path, model_dir)

    # 3. Build input/output signature (task-specific)
    import pandas as pd

    input_example = pd.DataFrame(
        [{"image": "base64_encoded_image_string"}]
    )

    if task_type == "classification":
        output_example = [
            {
                "predictions": {
                    "label": 0,
                    "label_name": "class_0",
                    "confidence": 0.95,
                    "top_k": [{"label": 0, "label_name": "class_0", "confidence": 0.95}],
                    "status": "success",
                }
            }
        ]
    elif task_type == "segmentation":
        output_example = [
            {
                "predictions": {
                    "segmentation_map": [[0, 1], [1, 0]],
                    "unique_classes": [0, 1],
                    "num_classes": 2,
                    "height": 2,
                    "width": 2,
                    "status": "success",
                }
            }
        ]
    else:
        output_example = [
            {
                "predictions": {
                    "boxes": [[0.0, 0.0, 100.0, 100.0]],
                    "scores": [0.95],
                    "labels": [1],
                    "num_detections": 1,
                    "status": "success",
                }
            }
        ]

    signature = infer_signature(input_example, output_example)

    # 4. Log the PyFunc model
    if task_type == "classification":
        from .pyfunc import ClassificationPyFuncModel

        pyfunc_model = ClassificationPyFuncModel()
        artifact_name = "classification_pyfunc"
    elif task_type == "segmentation":
        from .pyfunc import SegmentationPyFuncModel

        pyfunc_model = SegmentationPyFuncModel()
        artifact_name = "segmentation_pyfunc"
    else:
        from .pyfunc import DetectionPyFuncModel

        pyfunc_model = DetectionPyFuncModel()
        artifact_name = "detection_pyfunc"

    pip_requirements = [
        "mlflow>=3.1",
        "torch>=2.0",
        "transformers>=4.36",
        "Pillow>=9.0",
        "numpy>=1.24",
    ]

    model_info = mlflow.pyfunc.log_model(
        name=artifact_name,
        python_model=pyfunc_model,
        artifacts={"model_dir": model_dir},
        pip_requirements=pip_requirements,
        signature=signature,
        input_example=input_example,
    )

    pyfunc_model_uri = model_info.model_uri

    # 5. Validate with a real prediction (optional)
    if validate and test_image_path:
        print("Validating model with test image...")
        import base64

        with open(test_image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        test_input = pd.DataFrame([{"image": b64}])
        loaded = mlflow.pyfunc.load_model(pyfunc_model_uri)
        result = loaded.predict(test_input)
        print(f"Validation result: {result[0]['predictions']['status']}")
        assert result[0]["predictions"]["status"] == "success", "Validation failed"

    # 6. Register to Unity Catalog
    mv = mlflow.register_model(pyfunc_model_uri, registered_model_name)
    version = mv.version

    # 7. Set aliases and tags
    client = mlflow.MlflowClient()
    for alias in aliases:
        client.set_registered_model_alias(registered_model_name, alias, version)

    for k, v in tags.items():
        client.set_model_version_tag(registered_model_name, version, k, v)

    print(f"Registered {registered_model_name} version {version}")
    return {
        "model_uri": pyfunc_model_uri,
        "model_version": version,
        "registered_model_name": registered_model_name,
    }
