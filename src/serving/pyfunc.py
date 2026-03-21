"""PyFunc wrappers for Databricks Model Serving — detection and classification."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

import mlflow


# ======================================================================
# Shared base class
# ======================================================================

class _BaseCVPyFuncModel(mlflow.pyfunc.PythonModel):
    """Shared input normalisation and image loading for all CV tasks."""

    def _normalize_input(self, model_input: Any) -> List[Any]:
        """Convert any Model Serving input format to a list of records."""
        import pandas as pd

        if isinstance(model_input, pd.DataFrame):
            return model_input.to_dict(orient="records")

        if isinstance(model_input, dict):
            if "instances" in model_input:
                return model_input["instances"]
            if "inputs" in model_input:
                return model_input["inputs"]
            if "dataframe_records" in model_input:
                return model_input["dataframe_records"]
            if "dataframe_split" in model_input:
                cols = model_input["dataframe_split"]["columns"]
                data = model_input["dataframe_split"]["data"]
                return [dict(zip(cols, row)) for row in data]
            return [model_input]

        if isinstance(model_input, list):
            return model_input

        return [model_input]

    @staticmethod
    def _load_image(record: Any) -> Image.Image:
        """Load an image from base64, URL, numpy array, or raw dict."""
        if isinstance(record, Image.Image):
            return record.convert("RGB")

        if isinstance(record, np.ndarray):
            return Image.fromarray(record).convert("RGB")

        if isinstance(record, dict):
            if "image" in record:
                return _BaseCVPyFuncModel._decode_image(record["image"])
            if "b64" in record:
                return _BaseCVPyFuncModel._decode_image(record["b64"])
            if "url" in record:
                return _BaseCVPyFuncModel._load_from_url(record["url"])
            if "data" in record:
                return _BaseCVPyFuncModel._decode_image(record["data"])

        if isinstance(record, str):
            try:
                return _BaseCVPyFuncModel._decode_image(record)
            except Exception:
                pass
            return Image.open(record).convert("RGB")

        if isinstance(record, bytes):
            return Image.open(io.BytesIO(record)).convert("RGB")

        raise ValueError(f"Cannot load image from input type: {type(record)}")

    @staticmethod
    def _decode_image(b64_string: str) -> Image.Image:
        image_bytes = base64.b64decode(b64_string)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    @staticmethod
    def _load_from_url(url: str) -> Image.Image:
        import urllib.request

        with urllib.request.urlopen(url) as resp:
            image_bytes = resp.read()
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")


# ======================================================================
# Detection
# ======================================================================

class DetectionPyFuncModel(_BaseCVPyFuncModel):
    """Wraps a HuggingFace object detection model for Databricks Model Serving."""

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    DEFAULT_MAX_DETECTIONS = 100

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        model_dir = context.artifacts["model_dir"]
        self.model = AutoModelForObjectDetection.from_pretrained(model_dir)
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model.eval()

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        params = params or {}
        confidence_threshold = float(
            params.get("confidence_threshold", self.DEFAULT_CONFIDENCE_THRESHOLD)
        )
        max_detections = int(
            params.get("max_detections", self.DEFAULT_MAX_DETECTIONS)
        )

        records = self._normalize_input(model_input)
        results = []
        for record in records:
            try:
                pred = self._predict_single(record, confidence_threshold, max_detections)
                results.append({"predictions": {**pred, "status": "success"}})
            except Exception as e:
                results.append({"predictions": {"status": "error", "error": str(e)}})
        return results

    def _predict_single(
        self,
        record: Any,
        confidence_threshold: float,
        max_detections: int,
    ) -> Dict[str, Any]:
        image = self._load_image(record)

        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=confidence_threshold,
            target_sizes=target_sizes,
        )[0]

        boxes = results["boxes"].numpy().tolist()
        scores = results["scores"].numpy().tolist()
        labels = results["labels"].numpy().tolist()

        if len(scores) > max_detections:
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
                :max_detections
            ]
            boxes = [boxes[i] for i in indices]
            scores = [scores[i] for i in indices]
            labels = [labels[i] for i in indices]

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "num_detections": len(scores),
        }


# ======================================================================
# Classification
# ======================================================================

# ======================================================================
# Segmentation
# ======================================================================

class SegmentationPyFuncModel(_BaseCVPyFuncModel):
    """Wraps a HuggingFace segmentation model for Databricks Model Serving.

    Returns per-pixel class maps as nested lists (JSON-serialisable).
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        from transformers import AutoImageProcessor

        model_dir = context.artifacts["model_dir"]
        self.processor = AutoImageProcessor.from_pretrained(model_dir)

        # Determine model type by trying universal first, then semantic
        try:
            from transformers import AutoModelForUniversalSegmentation
            self.model = AutoModelForUniversalSegmentation.from_pretrained(model_dir)
            self.model_type = "universal"
        except Exception:
            from transformers import AutoModelForSemanticSegmentation
            self.model = AutoModelForSemanticSegmentation.from_pretrained(model_dir)
            self.model_type = "semantic"

        self.model.eval()

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        records = self._normalize_input(model_input)
        results = []
        for record in records:
            try:
                pred = self._predict_single(record)
                results.append({"predictions": {**pred, "status": "success"}})
            except Exception as e:
                results.append({"predictions": {"status": "error", "error": str(e)}})
        return results

    def _predict_single(self, record: Any) -> Dict[str, Any]:
        image = self._load_image(record)
        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = [image.size[::-1]]  # (H, W)

        if self.model_type == "universal":
            seg_maps = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=target_sizes,
            )
            seg_map = seg_maps[0].numpy()
        else:
            logits = outputs.logits
            upsampled = torch.nn.functional.interpolate(
                logits, size=target_sizes[0], mode="bilinear", align_corners=False,
            )
            seg_map = upsampled.argmax(dim=1)[0].numpy()

        unique_classes = np.unique(seg_map).tolist()
        return {
            "segmentation_map": seg_map.tolist(),
            "unique_classes": unique_classes,
            "num_classes": len(unique_classes),
            "height": seg_map.shape[0],
            "width": seg_map.shape[1],
        }


# ======================================================================
# Classification
# ======================================================================

class ClassificationPyFuncModel(_BaseCVPyFuncModel):
    """Wraps a HuggingFace image classification model for Databricks Model Serving."""

    DEFAULT_TOP_K = 5

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        model_dir = context.artifacts["model_dir"]
        self.model = AutoModelForImageClassification.from_pretrained(model_dir)
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model.eval()

        # Load id2label mapping if available
        self.id2label = getattr(self.model.config, "id2label", None) or {}

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        params = params or {}
        top_k = int(params.get("top_k", self.DEFAULT_TOP_K))

        records = self._normalize_input(model_input)
        results = []
        for record in records:
            try:
                pred = self._predict_single(record, top_k)
                results.append({"predictions": {**pred, "status": "success"}})
            except Exception as e:
                results.append({"predictions": {"status": "error", "error": str(e)}})
        return results

    def _predict_single(self, record: Any, top_k: int) -> Dict[str, Any]:
        image = self._load_image(record)

        inputs = self.processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)

        top_k = min(top_k, len(probs))
        topk_probs, topk_indices = probs.topk(top_k)

        top_predictions = []
        for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
            label_name = self.id2label.get(idx, str(idx))
            top_predictions.append({
                "label": idx,
                "label_name": label_name,
                "confidence": prob,
            })

        best = top_predictions[0]
        return {
            "label": best["label"],
            "label_name": best["label_name"],
            "confidence": best["confidence"],
            "top_k": top_predictions,
        }
