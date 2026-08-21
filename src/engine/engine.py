"""TrainingEngine — unified API that wires task, model, data, and trainer.

Single-node multi-GPU goes through ``TorchDistributor(local_mode=True)``, which
launches one process per GPU on the driver so HF Trainer runs real DDP.  Plain
``python jobs/train.py`` without a launcher would give DataParallel instead —
one process driving every GPU — which is both slower and has different
effective-batch semantics than the config implies.

Multi-node uses ``TorchDistributor(local_mode=False)`` to spread processes
across Spark workers.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Literal, Optional

import torch
from transformers import TrainingArguments

from ..config.schema import PipelineConfig
from ..registry import TaskRegistry
from ..serving.artifacts import LOGGED_MODEL_PARAM, log_model_artifacts
from ..utils.environment import (
    get_gpu_count,
    is_rank_zero,
    resolve_precision,
    setup_nccl_env,
    stage_data_to_local,
)
from .callbacks import EarlyStoppingCallback, VolumeCheckpointCallback
from .trainer import CVTrainer


class TrainingEngine:
    """High-level orchestrator: config in → metrics out."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(
        self,
        num_gpus: Optional[int] = None,
        distributed_mode: Literal["auto", "single", "local", "multinode"] = "auto",
    ) -> Dict[str, Any]:
        """Run training.

        Args:
            num_gpus: Number of GPUs to use.  Auto-detected when ``None``.
            distributed_mode:
                ``"auto"`` (default) — single process on one GPU, or
                    ``TorchDistributor(local_mode=True)`` when several GPUs are
                    visible on this node.
                ``"single"`` — force one process, even with several GPUs.
                ``"local"`` — force single-node multi-process DDP.
                ``"multinode"`` — distribute across Spark workers.
        """
        if num_gpus is None:
            num_gpus = get_gpu_count()
        num_gpus = max(num_gpus, 1)

        # Already inside a launched worker: just train, one process one GPU.
        if os.environ.get("WORLD_SIZE") and os.environ.get("RANK") is not None:
            return self._train_fn(num_gpus=1)

        if distributed_mode == "auto":
            distributed_mode = "local" if num_gpus > 1 else "single"

        if distributed_mode == "single":
            if num_gpus > 1:
                # Pin to one GPU so HF Trainer cannot silently pick DataParallel.
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
            return self._train_fn(num_gpus=1)

        # Multi-process paths need the data on local disk: /Volumes is a FUSE
        # mount the worker processes cannot all read efficiently.
        self._stage_volumes_data()
        return self._train_distributed(num_gpus, local_mode=(distributed_mode == "local"))

    # ------------------------------------------------------------------
    # Core training (one process, one device)
    # ------------------------------------------------------------------
    def _train_fn(self, num_gpus: int = 1) -> Dict[str, Any]:
        """Core training logic for a single process."""
        import mlflow
        from transformers import set_seed

        config = self.config
        is_writer = is_rank_zero()

        set_seed(config.training.seed)

        # --- task registry ---
        import src.tasks.detection  # noqa: F401  (triggers @register)
        import src.tasks.classification  # noqa: F401
        import src.tasks.segmentation  # noqa: F401

        task = TaskRegistry.get(config.model.task_type)

        # --- model + the processor that must ship with it ---
        model = task.get_model(config.model)
        processor = task.get_processor(config.model)

        # --- datasets ---
        train_ds = task.get_train_dataset(config)
        val_ds = task.get_val_dataset(config)

        # --- NCCL env for Databricks networking ---
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            setup_nccl_env()

        # --- optimizer + scheduler ---
        # Under DDP each rank sees 1/world_size of the data, so the number of
        # optimizer steps shrinks accordingly.  Computing this from the full
        # dataset would stretch the cosine schedule over a horizon the run
        # never reaches, leaving the LR high at the end of training.
        steps_per_epoch = math.ceil(
            len(train_ds) / (config.data.batch_size * world_size)
        )
        num_training_steps = max(steps_per_epoch * config.training.max_epochs, 1)
        optimizer, scheduler = task.create_optimizer_and_scheduler(
            model, config, num_training_steps,
        )

        # --- HF TrainingArguments ---
        output_dir = config.training.checkpoint_dir
        os.makedirs(output_dir, exist_ok=True)

        precision = resolve_precision(config.training.precision)

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
            metric_for_best_model="eval_{}".format(
                config.training.monitor_metric.removeprefix("val_").removeprefix("eval_")
            ),
            greater_is_better=(config.training.monitor_mode == "max"),
            report_to="mlflow",
            run_name=config.mlflow.run_name,
            bf16=(precision == "bf16"),
            fp16=(precision == "fp16"),
            seed=config.training.seed,
            data_seed=config.training.seed,
            remove_unused_columns=False,
            save_total_limit=config.training.save_top_k + 1,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=config.data.num_workers > 0,
        )

        # --- MLflow: single run for the entire lifecycle ---
        # Opening the run here ensures HF Trainer's MLflowCallback reuses it
        # (it checks mlflow.active_run() and sets _auto_end_run=False).
        # This prevents the duplicate-run problem where metrics and model
        # artifacts end up in different runs.
        mlflow.set_experiment(config.mlflow.experiment_name)

        run_ctx = mlflow.start_run(run_name=config.mlflow.run_name) if is_writer else None
        try:
            if is_writer:
                if config.mlflow.tags:
                    mlflow.set_tags(config.mlflow.tags)
                mlflow.log_params({
                    "task_type": config.model.task_type,
                    "model_name": config.model.model_name,
                    "num_classes": config.model.num_classes,
                    "max_epochs": config.training.max_epochs,
                    "batch_size": config.data.batch_size,
                    "learning_rate": config.model.learning_rate,
                    "num_gpus": num_gpus,
                    "world_size": world_size,
                    "seed": config.training.seed,
                    "precision": precision,
                })

            # --- callbacks ---
            callbacks = []
            if config.training.volume_checkpoint_dir:
                callbacks.append(
                    VolumeCheckpointCallback(
                        config.training.volume_checkpoint_dir,
                        save_total_limit=config.training.save_top_k + 1,
                    )
                )
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
            if hasattr(task, "compute_loss"):
                trainer.loss_fn = task.compute_loss
            if hasattr(task, "get_eval_fn"):
                trainer.eval_fn = task.get_eval_fn(config.model)

            # --- train ---
            trainer.train()

            # --- final eval ---
            metrics = trainer.evaluate()

            # --- log final model + processor (rank 0, inside the same run) ---
            # Not wrapped in a try/except: a run that trained but persisted no
            # model has failed, and should say so rather than exiting cleanly.
            if is_writer and config.mlflow.log_model:
                model_uri = log_model_artifacts(
                    trainer.model,
                    processor,
                    task_type=config.model.task_type,
                )
                mlflow.log_param(LOGGED_MODEL_PARAM, model_uri)
            if is_writer and config.training.volume_checkpoint_dir:
                mlflow.log_param("checkpoint_dir", config.training.volume_checkpoint_dir)

        finally:
            if run_ctx is not None:
                mlflow.end_run()

        return metrics

    # ------------------------------------------------------------------
    # TorchDistributor paths
    # ------------------------------------------------------------------
    def _train_distributed(self, num_gpus: int, local_mode: bool) -> Dict[str, Any]:
        """Launch one process per GPU via TorchDistributor.

        ``local_mode=True`` keeps every process on the driver node, which is
        what a single-node multi-GPU cluster needs.  ``local_mode=False``
        spreads them over Spark workers for multi-node runs.
        """
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
            return engine._train_fn(num_gpus=1)  # each worker process uses 1 GPU

        from pyspark.ml.torch.distributor import TorchDistributor

        return TorchDistributor(
            num_processes=num_gpus,
            local_mode=local_mode,
            use_gpu=True,
        ).run(train_fn)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _stage_volumes_data(self) -> None:
        """Stage /Volumes/ paths to local disk so worker processes can read them."""
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
