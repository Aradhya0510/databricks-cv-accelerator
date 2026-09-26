"""TrainingEngine — unified API that wires task, model, data, and trainer.

Runs on Databricks AI Runtime.  How the training processes come to exist
depends on where :meth:`TrainingEngine.train` is called from:

* **Inside a launched worker** (``LOCAL_RANK`` set by ``torchrun`` or
  ``@distributed``): train in this process on its own GPU.
* **One GPU**: train in this process.
* **Several GPUs, from a notebook**: fan out with ``serverless_gpu``'s
  ``@distributed``, one process per GPU.

Scripts do not take the last path: ``jobs/train.py`` relaunches itself under
``torchrun`` before it gets here, see :mod:`src.engine.launch`.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

from transformers import TrainingArguments

from ..config.schema import PipelineConfig
from ..registry import TaskRegistry
from ..serving.artifacts import LOGGED_MODEL_PARAM, log_model_artifacts
from ..utils.environment import (
    get_gpu_count,
    is_distributed_worker,
    is_rank_zero,
    resolve_precision,
    stage_data_to_local,
)
from .callbacks import EarlyStoppingCallback, VolumeCheckpointCallback, find_latest_checkpoint
from ..utils.manifest import write_run_manifest
from .trainer import CVTrainer

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


class TrainingEngine:
    """High-level orchestrator: config in → metrics out."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, num_gpus: Optional[int] = None) -> Dict[str, Any]:
        """Run training.

        Args:
            num_gpus: GPUs to train on.  Defaults to every GPU on the node.
        """
        if is_distributed_worker():
            return self._train_fn()

        if num_gpus is None:
            num_gpus = get_gpu_count()
        num_gpus = max(num_gpus, 1)

        self.stage_data()

        if num_gpus == 1:
            if get_gpu_count() > 1:
                # Pin to one GPU so HF Trainer cannot silently pick DataParallel.
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
            return self._train_fn()

        return self._train_serverless_gpu(num_gpus)

    def stage_data(self) -> None:
        """Point the data paths at local copies of any ``/Volumes`` inputs.

        Call once per node, before worker processes start.
        """
        data = self.config.data
        if not data.stage_to_local:
            return
        for field in (
            "train_data_path", "val_data_path", "test_data_path",
            "train_annotation_file", "val_annotation_file", "test_annotation_file",
        ):
            path = getattr(data, field)
            if path:
                setattr(data, field, stage_data_to_local(path, data.local_cache_dir))

    # ------------------------------------------------------------------
    # Core training (one process, one device)
    # ------------------------------------------------------------------
    def _train_fn(self) -> Dict[str, Any]:
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

        world_size = int(os.environ.get("WORLD_SIZE", "1"))

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

        resume_from = self._resolve_resume_checkpoint()

        # --- MLflow: single run for the entire lifecycle ---
        # Opening the run here ensures HF Trainer's MLflowCallback reuses it
        # (it checks mlflow.active_run() and sets _auto_end_run=False).
        # This prevents the duplicate-run problem where metrics and model
        # artifacts end up in different runs.
        run_ctx = self._start_mlflow_run() if is_writer else None
        status = "FAILED"
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
                    "world_size": world_size,
                    "seed": config.training.seed,
                    "precision": precision,
                })
                if resume_from:
                    mlflow.set_tag("resumed_from_checkpoint", resume_from)

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
            trainer.train(resume_from_checkpoint=resume_from)

            # --- final eval ---
            metrics = trainer.evaluate()

            # --- log final model + processor (rank 0, inside the same run) ---
            # Not wrapped in a try/except: a run that trained but persisted no
            # model has failed, and should say so rather than exiting cleanly.
            model_uri = None
            if is_writer and config.mlflow.log_model:
                model_uri = log_model_artifacts(
                    trainer.model,
                    processor,
                    task_type=config.model.task_type,
                )
                mlflow.log_param(LOGGED_MODEL_PARAM, model_uri)
            if is_writer and config.training.volume_checkpoint_dir:
                mlflow.log_param("checkpoint_dir", config.training.volume_checkpoint_dir)

            if is_writer:
                run = mlflow.active_run()
                path = write_run_manifest(
                    config.output.results_dir,
                    run_id=run.info.run_id,
                    experiment_id=run.info.experiment_id,
                    model_uri=model_uri,
                    task_type=config.model.task_type,
                )
                print(f"Run manifest written to {path}")

            status = "FINISHED"
        finally:
            if run_ctx is not None:
                mlflow.end_run(status=status)

        return metrics

    def _start_mlflow_run(self):
        """Open the run this process logs to.

        ``databricks air run``, ``ai_runtime_task`` and ``@distributed`` each
        create the MLflow run themselves and hand it over as ``MLFLOW_RUN_ID``.
        Logging anywhere else would split one training job across two runs, so
        that run wins over ``mlflow.experiment_name`` in the config.
        """
        import mlflow

        # Popped, not read: HF Trainer's MLflowCallback calls start_run whenever
        # MLFLOW_RUN_ID is set, even with that run already active, and MLflow
        # raises. With it gone the callback just reuses the active run.
        platform_run_id = os.environ.pop("MLFLOW_RUN_ID", None)
        if platform_run_id:
            print(f"Logging to the AI Runtime MLflow run {platform_run_id}")
            return mlflow.start_run(run_id=platform_run_id)

        mlflow.set_experiment(self.config.mlflow.experiment_name)
        return mlflow.start_run(run_name=self.config.mlflow.run_name)

    def _resolve_resume_checkpoint(self) -> Optional[str]:
        requested = self.config.training.resume_from_checkpoint
        if requested != "latest":
            return requested

        latest = find_latest_checkpoint(self.config.training.volume_checkpoint_dir)
        print(f"Resuming from {latest}" if latest else "No checkpoint to resume from; starting fresh")
        return latest

    # ------------------------------------------------------------------
    # Notebook multi-GPU
    # ------------------------------------------------------------------
    def _train_serverless_gpu(self, num_gpus: int) -> Dict[str, Any]:
        """Run one training process per GPU with ``serverless_gpu``'s ``@distributed``."""
        try:
            from serverless_gpu import distributed
        except ImportError:
            raise RuntimeError(
                f"Training on {num_gpus} GPUs needs one process per GPU. From a "
                f"notebook, attach it to AI Runtime so serverless_gpu is "
                f"available; from a script, run jobs/train.py, which relaunches "
                f"itself under torchrun."
            ) from None

        import mlflow

        # @distributed creates the run; this is the only way to choose where.
        mlflow.set_experiment(self.config.mlflow.experiment_name)

        config_dict = self.config.model_dump()
        project_root = _PROJECT_ROOT

        # Defined here rather than at module level so cloudpickle ships it by
        # value: the workers can only import ``src`` once this body has put the
        # project root on sys.path.
        def train_fn():
            import sys

            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from src.config.schema import PipelineConfig
            from src.engine.engine import TrainingEngine

            return TrainingEngine(PipelineConfig(**config_dict))._train_fn()

        # The decorator's 3-hour default would kill most real fine-tuning runs.
        results = distributed(gpus=num_gpus, timeout=None)(train_fn).distributed()
        if isinstance(results, (list, tuple)):
            return next((r for r in results if r is not None), {})
        return results
