"""Computer Vision Tasks

This package contains implementations for various computer vision tasks.

Note: Task registration with the TaskRegistry happens lazily when
the engine imports ``src.tasks.detection`` (or when you import it directly).
This avoids pulling in heavy dependencies (transformers, torch) at package init.
"""
