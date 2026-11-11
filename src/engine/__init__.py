"""Engine module — HF Trainer training for Databricks AI Runtime.

``TrainingEngine`` is resolved lazily so that ``src.engine.launch`` stays
importable without torch and transformers.
"""

__all__ = ["TrainingEngine"]


def __getattr__(name):
    if name == "TrainingEngine":
        from .engine import TrainingEngine

        return TrainingEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
