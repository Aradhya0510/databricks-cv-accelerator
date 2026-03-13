"""Task registry with ``@register`` decorator.

Each registered task class must provide methods that let the engine
build a complete training run without knowing task-specific details.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type


class TaskRegistry:
    """Central registry of CV task implementations."""

    _tasks: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """Class decorator that registers a task under *name*."""

        def decorator(task_cls: Type) -> Type:
            cls._tasks[name] = task_cls
            return task_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Any:
        """Return an *instance* of the task registered under *name*."""
        if name not in cls._tasks:
            available = ", ".join(cls._tasks.keys()) or "(none)"
            raise KeyError(f"Unknown task '{name}'. Available: {available}")
        return cls._tasks[name]()

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._tasks.keys())
