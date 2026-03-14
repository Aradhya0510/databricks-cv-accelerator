"""Configuration module with Pydantic v2 schema and YAML loader."""

from .schema import PipelineConfig, load_config

__all__ = ["PipelineConfig", "load_config"]
