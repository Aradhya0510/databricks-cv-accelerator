"""Attach human-readable class names to a model config.

``id2label`` / ``label2id`` ride along with ``save_pretrained``, so setting
them once at model construction is all it takes for the serving wrappers to
return real class names instead of ``LABEL_0``.  Without this the label
information is known at training time and then thrown away.
"""

from __future__ import annotations

from typing import List, Optional


def apply_label_names(
    model,
    class_names: Optional[List[str]],
    num_classes: int,
) -> None:
    """Set ``id2label`` / ``label2id`` on *model*'s config, in place.

    Does nothing when *class_names* is absent, leaving whatever the checkpoint
    shipped with.  When present it must already match *num_classes* — the
    config schema enforces that, so a mismatch here is a programming error.
    """
    if not class_names:
        return

    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names has {len(class_names)} entries but num_classes is "
            f"{num_classes}"
        )

    model.config.id2label = {i: name for i, name in enumerate(class_names)}
    model.config.label2id = {name: i for i, name in enumerate(class_names)}


def label_names_from_config(model_config, num_classes: int) -> List[str]:
    """Read class names back off a model config, falling back to indices."""
    id2label = getattr(model_config, "id2label", None) or {}
    return [str(id2label.get(i, i)) for i in range(num_classes)]
