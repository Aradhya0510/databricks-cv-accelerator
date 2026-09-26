"""HF Trainer callbacks for checkpoint management and early stopping."""

from __future__ import annotations

import os
import re
import shutil
from typing import Optional

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers import EarlyStoppingCallback as _HFEarlyStoppingCallback

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


class VolumeCheckpointCallback(TrainerCallback):
    """Copy HF Trainer checkpoints to a persistent /Volumes/ directory.

    Training writes checkpoints to fast local disk; this mirrors them onto a
    UC Volume so they survive the cluster.

    Two things this has to get right that a naive mirror does not:

    * **Only rank 0 copies.**  Under DDP every rank runs ``on_save``, and
      without this guard they would all write the same checkpoint to the same
      Volume path concurrently.
    * **Retention is mirrored.**  ``save_total_limit`` prunes the local
      directory but says nothing about the Volume, so the mirror used to grow
      without bound — tens of GB per epoch for a large model.
    * **A checkpoint appears whole or not at all.**  Each copy lands in a
      ``.partial`` directory and is renamed into place, so a run killed
      mid-copy never leaves a truncated ``checkpoint-N`` for
      :func:`find_latest_checkpoint` to resume from.
    """

    def __init__(self, volume_dir: str, save_total_limit: Optional[int] = None):
        self.volume_dir = volume_dir
        self.save_total_limit = save_total_limit
        os.makedirs(self.volume_dir, exist_ok=True)
        print(f"Checkpoints will be copied to volume: {self.volume_dir}")

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            return

        dest = os.path.join(self.volume_dir, f"checkpoint-{state.global_step}")
        partial = f"{dest}.partial"
        try:
            for stale in (partial, dest):
                if os.path.exists(stale):
                    shutil.rmtree(stale)
            shutil.copytree(ckpt_dir, partial)
            os.rename(partial, dest)
            print(f"Copied checkpoint to volume: {dest}")
        except Exception as e:
            print(f"Warning: failed to copy checkpoint to volume: {e}")
            return

        self._prune()

    def _prune(self) -> None:
        """Keep only the newest ``save_total_limit`` checkpoints on the Volume."""
        if not self.save_total_limit or self.save_total_limit <= 0:
            return

        try:
            checkpoints = []
            for name in os.listdir(self.volume_dir):
                match = _CHECKPOINT_RE.match(name)
                if match and os.path.isdir(os.path.join(self.volume_dir, name)):
                    checkpoints.append((int(match.group(1)), name))

            checkpoints.sort()
            for _, name in checkpoints[: -self.save_total_limit]:
                shutil.rmtree(os.path.join(self.volume_dir, name), ignore_errors=True)
                print(f"Pruned old volume checkpoint: {name}")
        except Exception as e:
            print(f"Warning: failed to prune volume checkpoints: {e}")


def find_latest_checkpoint(checkpoint_dir: Optional[str]) -> Optional[str]:
    """Return the newest complete ``checkpoint-N`` under *checkpoint_dir*, if any.

    Complete means HF Trainer's ``trainer_state.json`` is present; resuming
    from a directory without it fails deep inside ``Trainer.train``.
    """
    if not checkpoint_dir or not os.path.isdir(checkpoint_dir):
        return None

    candidates = []
    for name in os.listdir(checkpoint_dir):
        match = _CHECKPOINT_RE.match(name)
        path = os.path.join(checkpoint_dir, name)
        if match and os.path.isfile(os.path.join(path, "trainer_state.json")):
            candidates.append((int(match.group(1)), path))

    return max(candidates)[1] if candidates else None


class EarlyStoppingCallback(_HFEarlyStoppingCallback):
    """Thin wrapper around HF's built-in EarlyStoppingCallback with sensible defaults."""

    def __init__(self, early_stopping_patience: int = 10, early_stopping_threshold: float = 0.0):
        super().__init__(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
