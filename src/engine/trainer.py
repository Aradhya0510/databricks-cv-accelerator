"""Generic CV Trainer that delegates task-specific logic to the task definition."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput


class CVTrainer(Trainer):
    """``transformers.Trainer`` subclass that delegates task-specific
    loss computation and evaluation to callable hooks.

    Attributes:
        loss_fn:  ``(model, inputs) -> loss`` or ``(model, inputs) -> (loss, outputs)``.
                  When *None*, falls back to ``model(**inputs).loss``.
        eval_fn:  ``(model, dataloader, args, metric_key_prefix) -> EvalLoopOutput``.
                  When *None*, falls back to the default HF Trainer evaluation.
    """

    loss_fn: Optional[Callable] = None
    eval_fn: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.loss_fn is not None:
            return self.loss_fn(model, inputs, return_outputs=return_outputs)

        # Default: assume model returns an object with .loss
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------
    # Evaluation loop override
    # ------------------------------------------------------------------
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        if self.eval_fn is not None:
            model = self._wrap_model(self.model, training=False, dataloader=dataloader)
            model.eval()
            return self.eval_fn(
                model=model,
                dataloader=dataloader,
                args=self.args,
                metric_key_prefix=metric_key_prefix,
            )

        # Default HF Trainer evaluation
        return super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
