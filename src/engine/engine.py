"""TrainingEngine — unified API that wires task, model, data, and trainer.

Single-node multi-GPU: native HF Trainer DDP (no Spark orchestration).
Multi-node: opt-in via TorchDistributor when ``distributed_mode="torchd"``.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Literal, Optional

import torch
from transformers import TrainingArguments

from ..config.schema import PipelineConfig
from ..registry import TaskRegistry
from ..utils.environment import get_gpu_count, setup_nccl_env, stage_data_to_local
from .callbacks import EarlyStoppingCallback, VolumeCheckpointCallback
from .trainer import CVTrainer


class TrainingEngine:
    """High-level orchestrator: config in → metrics out.

    By default, uses native HF Trainer DDP for multi-GPU on a single node.
    Pass ``distributed_mode="torchd"`` only when you need multi-node
    distribution via Spark's TorchDistributor.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(
        self,
        num_gpus: Optional[int] = None,
        distributed_mode: Literal["native", "torchd"] = "native",
    ) -> Dict[str, Any]:
        """Run training.

        Args:
            num_gpus: Number of GPUs to use.  Auto-detected when ``None``.
            distributed_mode:
                ``"native"`` (default) — let HF Trainer handle DDP directly.
                    Works for single-node multi-GPU without Spark overhead.
                ``"torchd"`` — use ``TorchDistributor`` for multi-node
                    distribution via Spark workers.
        """
        if num_gpus is None:
            num_gpus = get_gpu_count()
        num_gpus = max(num_gpus, 1)

        # Stage /Volumes/ data to /tmp/ so DDP workers can access it
        if num_gpus > 1:
            self._stage_volumes_data()

        if distributed_mode == "torchd" and num_gpus > 1:
            return self._train_torchd(num_gpus)

        # Native path: HF Trainer manages DDP internally
        return self._train_fn(num_gpus=num_gpus)

    # ------------------------------------------------------------------
    # Native training (single-GPU or single-node multi-GPU DDP)
    # ------------------------------------------------------------------
    def _train_fn(self, num_gpus: int = 1) -> Dict[str, Any]:
        """Core training logic.  HF Trainer handles DDP when num_gpus > 1."""
        config = self.config

        # --- task registry ---
        import src.tasks.detection  # noqa: F401  (triggers @register)
        import src.tasks.classification  # noqa: F401

        task = TaskRegistry.get(config.model.task_type)

        # --- model ---
        model = task.get_model(config.model)

        # --- datasets ---
        train_ds = task.get_train_dataset(config)
        val_ds = task.get_val_dataset(config)

        # --- optimizer + scheduler ---
        steps_per_epoch = math.ceil(len(train_ds) / config.data.batch_size)
        num_training_steps = steps_per_epoch * config.training.max_epochs
        optimizer, scheduler = task.create_optimizer_and_scheduler(
            model, config, num_training_steps,
        )

        # --- NCCL env for Databricks networking ---
        if num_gpus > 1:
            setup_nccl_env()

        # --- HF TrainingArguments ---
        output_dir = config.training.checkpoint_dir
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.training.max_epochs,
            per_device_train_batch_size=config.data.batch_size,
            per_device_eval_batch_size=config.data.batch_size,
            dataloader_num_workers=config.data.num_workers,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=config.training.log_every_n_steps,
            load_best_model_at_end=True,
            # monitor_metric may use Lightning-style "val_" prefix; strip it
            # since HF Trainer already adds "eval_" prefix to all eval metrics.
            metric_for_best_model="eval_{}".format(
                config.training.monitor_metric.removeprefix("val_")
            ),
            greater_is_better=(config.training.monitor_mode == "max"),
            report_to="mlflow",
            bf16=torch.cuda.is_available(),
            remove_unused_columns=False,
            save_total_limit=config.training.save_top_k + 1,  # +1 for "last"
            dataloader_pin_memory=True,
            dataloader_persistent_workers=config.data.num_workers > 0,
        )

        # --- MLflow setup ---
        try:
            import mlflow
            mlflow.set_experiment(config.mlflow.experiment_name)
        except Exception as e:
            print(f"Warning: MLflow experiment setup failed: {e}")

        # --- callbacks ---
        callbacks = []
        if config.training.volume_checkpoint_dir:
            callbacks.append(VolumeCheckpointCallback(config.training.volume_checkpoint_dir))
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config.training.early_stopping_patience)
        )

        # --- trainer ---
        trainer = CVTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=task.get_collate_fn(),
            optimizers=(optimizer, scheduler),
            callbacks=callbacks,
        )
        # Wire task-specific hooks into the generic trainer
        if hasattr(task, "compute_loss"):
            trainer.loss_fn = task.compute_loss
        if hasattr(task, "get_eval_fn"):
            trainer.eval_fn = task.get_eval_fn(config.model)

        # --- train ---
        trainer.train()

        # --- final eval ---
        metrics = trainer.evaluate()

        # --- log final model (rank 0 only) ---
        try:
            import mlflow
            if int(os.environ.get("LOCAL_RANK", "0")) == 0:
                model_info = mlflow.transformers.log_model(
                    transformers_model=model,
                    name="model",
                )
                mlflow.log_param("logged_model_uri", model_info.model_uri)
        except Exception as e:
            print(f"Warning: MLflow model logging failed: {e}")

        return metrics

    # ------------------------------------------------------------------
    # TorchDistributor path (multi-node only)
    # ------------------------------------------------------------------
    def _train_torchd(self, num_gpus: int) -> Dict[str, Any]:
        """Use TorchDistributor for multi-node distributed training."""
        config_dict = self.config.model_dump()

        def train_fn():
            import os
            os.environ.setdefault("NCCL_SOCKET_IFNAME", "eth0")
            os.environ.setdefault("NCCL_IB_DISABLE", "1")
            os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")
            os.environ.setdefault("NCCL_SHM_DISABLE", "1")

            from src.config.schema import PipelineConfig
            from src.engine.engine import TrainingEngine

            config = PipelineConfig(**config_dict)
            engine = TrainingEngine(config)
            return engine._train_fn(num_gpus=1)  # each worker uses 1 GPU

        from pyspark.ml.torch.distributor import TorchDistributor

        return TorchDistributor(
            num_processes=num_gpus,
            local_mode=False,  # multi-node: distribute across Spark workers
            use_gpu=True,
        ).run(train_fn)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _stage_volumes_data(self) -> None:
        """Stage /Volumes/ paths to /tmp/ so DDP workers can access them."""
        cfg = self.config
        cfg.data.train_data_path = stage_data_to_local(cfg.data.train_data_path)
        cfg.data.val_data_path = stage_data_to_local(cfg.data.val_data_path)
        if cfg.data.train_annotation_file:
            cfg.data.train_annotation_file = stage_data_to_local(cfg.data.train_annotation_file)
        if cfg.data.val_annotation_file:
            cfg.data.val_annotation_file = stage_data_to_local(cfg.data.val_annotation_file)
        if cfg.data.test_data_path:
            cfg.data.test_data_path = stage_data_to_local(cfg.data.test_data_path)
        if cfg.data.test_annotation_file:
            cfg.data.test_annotation_file = stage_data_to_local(cfg.data.test_annotation_file)
