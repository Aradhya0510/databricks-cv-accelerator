"""Model registration: save artifacts, log PyFunc, validate, register to UC."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional

import mlflow
from mlflow.models import infer_signature


def _resolve_model_artifacts(
    run_id: str,
    model_uri: Optional[str] = None,
) -> str:
    """Download the HF model artifacts logged during training.

    In MLflow 3.x, model artifacts live under a dedicated LoggedModel rather
    than as run artifacts.  When *model_uri* is provided (preferred) we use
    it directly.  Otherwise we search for a LoggedModel linked to *run_id*,
    falling back to the deprecated ``runs:/`` URI scheme.
    """
    if model_uri:
        return mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

    client = mlflow.MlflowClient()

    # Prefer the model_uri stored as a run param by the training engine
    run = client.get_run(run_id)
    stored_uri = run.data.params.get("logged_model_uri")
    if stored_uri:
        return mlflow.artifacts.download_artifacts(artifact_uri=stored_uri)

    # Fallback: runs:/ URI (deprecated in MLflow 3 but still functional)
    return mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model",
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
    from transformers import AutoImageProcessor

    aliases = aliases or ["champion", "latest"]
    tags = tags or {}

    # 1. Download model artifact
    artifact_path = _resolve_model_artifacts(run_id, model_uri)

    # 2. Save model + processor to a clean temp directory
    tmpdir = tempfile.mkdtemp(prefix="cv_pyfunc_")
    model_dir = os.path.join(tmpdir, "model_artifacts")

    if task_type == "classification":
        from transformers import AutoModelForImageClassification

        model = AutoModelForImageClassification.from_pretrained(artifact_path)
    elif task_type == "segmentation":
        try:
            from transformers import AutoModelForUniversalSegmentation
            model = AutoModelForUniversalSegmentation.from_pretrained(artifact_path)
        except Exception:
            from transformers import AutoModelForSemanticSegmentation
            model = AutoModelForSemanticSegmentation.from_pretrained(artifact_path)
    else:
        from transformers import AutoModelForObjectDetection

        model = AutoModelForObjectDetection.from_pretrained(artifact_path)

    processor = AutoImageProcessor.from_pretrained(artifact_path)
    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)

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
