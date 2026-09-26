"""
Databricks Client for Jobs API and MLflow Integration
Handles job submission, monitoring, and MLflow interactions
"""

import io
import os
import shlex
import tarfile
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat
import mlflow
from mlflow.tracking import MlflowClient


class DatabricksJobClient:
    """Client for Databricks Jobs API and MLflow."""
    
    def __init__(self):
        """Initialize Databricks client."""
        self.workspace_client = WorkspaceClient()
        mlflow.set_tracking_uri("databricks")
        # Three-level catalog.schema.model names are only valid against the
        # Unity Catalog registry; the workspace default may not be UC.
        mlflow.set_registry_uri("databricks-uc")
        self.mlflow_client = MlflowClient(tracking_uri="databricks")
    
    def create_training_job(
        self,
        job_name: str,
        config_path: str,
        project_path: str,
        experiment_name: str,
        accelerator_type: str = "GPU_1xA10",
        accelerator_count: int = 1,
        email_notifications: Optional[List[str]] = None,
    ) -> str:
        """
        Create an AI Runtime training job running the HF Trainer entry point.

        Args:
            job_name: Name for the job
            config_path: Path to configuration YAML (a /Volumes path, so the job can read it)
            project_path: Workspace path to the project root
                (e.g. /Workspace/Users/user@databricks.com/databricks-cv-accelerator)
            experiment_name: The config's mlflow.experiment_name. AI Runtime
                creates the run itself, so it is placed in this experiment for
                the dashboard to find.
            accelerator_type: GPU_1xA10, GPU_1xH100, GPU_8xH100 or GPU_8xB300
            accelerator_count: Total GPUs; a multiple of the per-node count
            email_notifications: Optional list of emails for notifications

        Returns:
            Job ID
        """
        # An ai_runtime_task runs a script and cannot pass it arguments, so the
        # config path is baked into a per-job script next to the project.
        jobs_dir = f"{project_path}/.air_jobs"
        command_path = f"{jobs_dir}/{job_name}.sh"
        script = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'cd "$CODE_SOURCE_PATH"\n'
            f"exec python jobs/train.py --config_path {shlex.quote(config_path)}\n"
        )
        self.workspace_client.workspace.mkdirs(jobs_dir)
        self.workspace_client.workspace.upload(
            command_path, io.BytesIO(script.encode()), format=ImportFormat.AUTO, overwrite=True,
        )
        code_path = self._package_project(project_path, f"{jobs_dir}/{job_name}.tgz")

        ai_runtime_task: Dict[str, Any] = {
            "code_source_path": code_path,
            "deployments": [{
                "name": "train",
                "command_path": command_path,
                "compute": {
                    "accelerator_type": accelerator_type,
                    "accelerator_count": accelerator_count,
                },
            }],
        }
        ai_runtime_task.update(self._ai_runtime_experiment(experiment_name))

        body: Dict[str, Any] = {
            "name": job_name,
            "max_concurrent_runs": 1,
            "tasks": [{
                "task_key": "train_model",
                "description": "Fine-tune CV model with HF Trainer on AI Runtime",
                "environment_key": "gpu",
                "ai_runtime_task": ai_runtime_task,
            }],
            "environments": [{
                "environment_key": "gpu",
                "spec": {
                    "base_environment": "databricks_ai_v6",
                    "dependencies": [f"-r {project_path}/requirements_runtime.txt"],
                },
            }],
        }
        if email_notifications:
            body["email_notifications"] = {
                "on_success": email_notifications,
                "on_failure": email_notifications,
            }

        # Raw REST rather than SDK dataclasses: the SDK preinstalled in the Apps
        # runtime can predate ai_runtime_task.
        created = self.workspace_client.api_client.do("POST", "/api/2.2/jobs/create", body=body)
        return str(created["job_id"])

    # What a training node needs from the project folder.
    _CODE_ENTRIES = ("src", "jobs", "configs", "requirements_runtime.txt")

    def _package_project(self, project_path: str, tarball_path: str) -> str:
        """Tar the project's code into *tarball_path* and return that path.

        ai_runtime_task's code_source_path must be a tarball — a folder fails
        with "Tarball not found".  Everything goes under one top-level
        directory because AI Runtime sets $CODE_SOURCE_PATH to the archive's
        first top-level entry, which then is the project root.
        """
        ws = self.workspace_client.workspace
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for entry in self._CODE_ENTRIES:
                root = f"{project_path}/{entry}"
                try:
                    info = ws.get_status(root)
                except Exception:
                    continue
                objects = [info] if info.object_type.value == "FILE" else ws.list(root, recursive=True)
                for obj in objects:
                    if obj.object_type.value != "FILE" or "__pycache__" in obj.path:
                        continue
                    data = ws.download(obj.path).read()
                    member = tarfile.TarInfo("project" + obj.path[len(project_path):])
                    member.size = len(data)
                    tar.addfile(member, io.BytesIO(data))
        buf.seek(0)
        ws.upload(tarball_path, buf, format=ImportFormat.AUTO, overwrite=True)
        return tarball_path

    @staticmethod
    def _ai_runtime_experiment(experiment_name: str) -> Dict[str, str]:
        """Map an MLflow experiment path onto ai_runtime_task's name + directory.

        ``/Users/me@x.com/cv`` becomes experiment ``cv`` under
        ``/Workspace/Users/me@x.com``, which is the same experiment.
        """
        experiment_name = experiment_name.rstrip("/")
        if not experiment_name.startswith("/"):
            return {"experiment": experiment_name}
        directory, name = experiment_name.rsplit("/", 1)
        if not directory.startswith("/Workspace"):
            directory = f"/Workspace{directory}"
        return {"experiment": name, "mlflow_experiment_directory": directory}
    
    def run_job(self, job_id: str, parameters: Optional[Dict[str, str]] = None) -> str:
        """
        Run a job.
        
        Args:
            job_id: Job ID to run
            parameters: Optional job parameters
            
        Returns:
            Run ID
        """
        if parameters:
            run = self.workspace_client.jobs.run_now(
                job_id=int(job_id),
                python_params=parameters
            )
        else:
            run = self.workspace_client.jobs.run_now(job_id=int(job_id))
        
        return str(run.run_id)
    
    @staticmethod
    def _clean_enum(value) -> str:
        """Strip SDK enum prefix: 'RunLifeCycleState.RUNNING' -> 'RUNNING'."""
        s = str(value)
        return s.rsplit(".", 1)[-1] if "." in s else s

    def get_job_status(self, run_id: str) -> Dict[str, Any]:
        """
        Get job run status.
        
        Args:
            run_id: Run ID to check
            
        Returns:
            Dictionary with status information
        """
        run = self.workspace_client.jobs.get_run(run_id=int(run_id))
        
        state = run.state
        life_cycle_state = self._clean_enum(state.life_cycle_state) if state else "UNKNOWN"
        result_state = self._clean_enum(state.result_state) if state and state.result_state else "UNKNOWN"
        state_message = state.state_message if state else ""
        
        # Get timing information
        start_time = None
        end_time = None
        duration_seconds = None
        
        if run.start_time:
            start_time = datetime.fromtimestamp(run.start_time / 1000)
        if run.end_time:
            end_time = datetime.fromtimestamp(run.end_time / 1000)
        if start_time and end_time:
            duration_seconds = (end_time - start_time).total_seconds()
        
        return {
            "run_id": run_id,
            "life_cycle_state": life_cycle_state,
            "result_state": result_state,
            "state_message": state_message,
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": duration_seconds,
            "run_page_url": run.run_page_url,
        }
    
    def cancel_job(self, run_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            run_id: Run ID to cancel
            
        Returns:
            True if cancelled successfully
        """
        try:
            self.workspace_client.jobs.cancel_run(run_id=int(run_id))
            return True
        except Exception as e:
            print(f"Error cancelling job: {e}")
            return False
    
    def get_mlflow_experiments(self) -> List[Dict[str, Any]]:
        """
        Get list of MLflow experiments.
        
        Returns:
            List of experiment dictionaries
        """
        experiments = self.mlflow_client.search_experiments()
        return [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "lifecycle_stage": exp.lifecycle_stage,
                "artifact_location": exp.artifact_location,
            }
            for exp in experiments
        ]
    
    def _resolve_experiment_id(self, experiment_name_or_id: str) -> Optional[str]:
        """Resolve an experiment name or numeric ID to an experiment ID."""
        # Try by name first
        exp = mlflow.get_experiment_by_name(experiment_name_or_id)
        if exp:
            return exp.experiment_id
        # If the value looks like a numeric ID, try direct lookup
        try:
            exp = self.mlflow_client.get_experiment(experiment_name_or_id)
            if exp:
                return exp.experiment_id
        except Exception:
            pass
        return None

    def get_mlflow_runs(
        self,
        experiment_name: str,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get MLflow runs for an experiment.
        
        Args:
            experiment_name: Experiment name (path) or numeric experiment ID
            max_results: Maximum number of runs to return
            
        Returns:
            List of run dictionaries
        """
        try:
            experiment_id = self._resolve_experiment_id(experiment_name)
            if not experiment_id:
                return []
            
            runs = self.mlflow_client.search_runs(
                experiment_ids=[experiment_id],
                max_results=max_results,
                order_by=["start_time DESC"]
            )
            
            return [
                {
                    "run_id": run.info.run_id,
                    "run_name": run.data.tags.get("mlflow.runName", "unnamed"),
                    "status": run.info.status,
                    "start_time": datetime.fromtimestamp(run.info.start_time / 1000) if run.info.start_time else None,
                    "end_time": datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "tags": run.data.tags,
                    "artifact_uri": run.info.artifact_uri,
                }
                for run in runs
            ]
        except Exception as e:
            print(f"Error fetching MLflow runs: {e}")
            return []
    
    def get_run_metrics_history(
        self,
        run_id: str,
        metric_key: str
    ) -> List[Dict[str, Any]]:
        """
        Get metric history for a run.
        
        Args:
            run_id: MLflow run ID
            metric_key: Metric name
            
        Returns:
            List of metric values with timestamps and steps
        """
        try:
            history = self.mlflow_client.get_metric_history(run_id, metric_key)
            return [
                {
                    "step": metric.step,
                    "value": metric.value,
                    "timestamp": datetime.fromtimestamp(metric.timestamp / 1000),
                }
                for metric in history
            ]
        except Exception as e:
            print(f"Error fetching metric history: {e}")
            return []
    
    def get_registered_models(
        self,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get list of registered models.
        
        Args:
            max_results: Maximum number of models to return
            
        Returns:
            List of registered model dictionaries
        """
        try:
            models = self.mlflow_client.search_registered_models(max_results=max_results)
            return [
                {
                    "name": model.name,
                    "creation_timestamp": datetime.fromtimestamp(model.creation_timestamp / 1000) if model.creation_timestamp else None,
                    "last_updated_timestamp": datetime.fromtimestamp(model.last_updated_timestamp / 1000) if model.last_updated_timestamp else None,
                    "description": model.description,
                    "latest_versions": [
                        {
                            "version": version.version,
                            "stage": getattr(version, "current_stage", None) or "N/A",
                            "aliases": getattr(version, "aliases", []) or [],
                            "run_id": version.run_id,
                        }
                        for version in (model.latest_versions or [])
                    ]
                }
                for model in models
            ]
        except Exception as e:
            print(f"Error fetching registered models: {e}")
            return []
    
    def create_model_serving_endpoint(
        self,
        endpoint_name: str,
        model_name: str,
        model_version: str,
        workload_size: str = "Small",
        scale_to_zero: bool = True,
    ) -> Dict[str, Any]:
        """
        Create or update a model serving endpoint.
        
        Args:
            endpoint_name: Name for the endpoint
            model_name: Registered model name
            model_version: Model version to serve
            workload_size: Size of the workload (Small, Medium, Large)
            scale_to_zero: Whether to enable scale-to-zero
            
        Returns:
            Endpoint information dictionary
        """
        from databricks.sdk.service.serving import (
            EndpointCoreConfigInput,
            ServedEntityInput,
        )
        
        try:
            # Check if endpoint exists
            try:
                existing_endpoint = self.workspace_client.serving_endpoints.get(endpoint_name)
                endpoint_exists = True
            except:
                endpoint_exists = False
            
            # Prepare configuration
            served_entity = ServedEntityInput(
                entity_name=model_name,
                entity_version=model_version,
                workload_size=workload_size,
                scale_to_zero_enabled=scale_to_zero
            )
            
            if endpoint_exists:
                # Update existing endpoint
                self.workspace_client.serving_endpoints.update_config(
                    name=endpoint_name,
                    served_entities=[served_entity]
                )
                status = "updated"
            else:
                # Create new endpoint
                config = EndpointCoreConfigInput(served_entities=[served_entity])
                self.workspace_client.serving_endpoints.create(
                    name=endpoint_name,
                    config=config
                )
                status = "created"
            
            return {
                "endpoint_name": endpoint_name,
                "status": status,
                "model_name": model_name,
                "model_version": model_version,
            }
        except Exception as e:
            return {
                "endpoint_name": endpoint_name,
                "status": "error",
                "error": str(e),
            }
    
    def get_endpoint_status(self, endpoint_name: str) -> Dict[str, Any]:
        """
        Get serving endpoint status including served models.
        
        Args:
            endpoint_name: Name of the endpoint
            
        Returns:
            Endpoint status dictionary
        """
        try:
            endpoint = self.workspace_client.serving_endpoints.get(endpoint_name)
            
            state = endpoint.state
            raw_config = str(state.config_update) if state else "UNKNOWN"
            raw_ready = str(state.ready) if state else "UNKNOWN"
            config_state = raw_config.split(".")[-1] if "." in raw_config else raw_config
            ready = raw_ready.split(".")[-1] if "." in raw_ready else raw_ready
            
            served_models = []
            if endpoint.config and endpoint.config.served_entities:
                for entity in endpoint.config.served_entities:
                    served_models.append({
                        "entity_name": entity.entity_name,
                        "entity_version": entity.entity_version,
                        "workload_size": getattr(entity, "workload_size", "N/A"),
                    })
            
            return {
                "endpoint_name": endpoint_name,
                "state": config_state,
                "ready": ready,
                "endpoint_url": getattr(endpoint, "url", None),
                "served_models": served_models,
            }
        except Exception as e:
            return {
                "endpoint_name": endpoint_name,
                "state": "NOT_FOUND",
                "error": str(e),
            }
    
    def query_endpoint(
        self,
        endpoint_name: str,
        image_bytes: bytes,
    ) -> Dict[str, Any]:
        """
        Send an image to a serving endpoint and return predictions.

        Args:
            endpoint_name: Serving endpoint name
            image_bytes: Raw image bytes (JPEG/PNG)

        Returns:
            Prediction result dictionary
        """
        import base64 as _b64

        encoded = _b64.b64encode(image_bytes).decode("utf-8")
        payload = {"dataframe_split": {"columns": ["image"], "data": [[encoded]]}}

        try:
            resp = self.workspace_client.serving_endpoints.query(
                name=endpoint_name,
                dataframe_split=payload["dataframe_split"],
            )
            return {"predictions": resp.predictions if hasattr(resp, "predictions") else resp.as_dict()}
        except Exception as e:
            return {"error": str(e)}
    
    # ------------------------------------------------------------------ #
    #  Unity Catalog Volume helpers (for Databricks App runtime)          #
    # ------------------------------------------------------------------ #

    def list_volume_files(self, volume_path: str, extensions: set = None) -> List[str]:
        """List files in a Unity Catalog Volume directory via the SDK.

        Args:
            volume_path: /Volumes/catalog/schema/volume/… path
            extensions: optional set of lowercase extensions to filter (e.g. {".jpg", ".png"})

        Returns:
            List of file names (not full paths).
        """
        try:
            entries = self.workspace_client.files.list_directory_contents(volume_path)
            names = []
            for entry in entries:
                name = entry.path.rstrip("/").rsplit("/", 1)[-1] if "/" in entry.path else entry.path
                if entry.is_directory:
                    continue
                if extensions and os.path.splitext(name)[1].lower() not in extensions:
                    continue
                names.append(name)
            return names
        except Exception as e:
            print(f"Error listing volume files at {volume_path}: {e}")
            return []

    def list_volume_dirs(self, volume_path: str) -> List[str]:
        """List sub-directories in a Volume path."""
        try:
            entries = self.workspace_client.files.list_directory_contents(volume_path)
            return [
                entry.path.rstrip("/").rsplit("/", 1)[-1]
                for entry in entries if entry.is_directory
            ]
        except Exception as e:
            print(f"Error listing volume dirs at {volume_path}: {e}")
            return []

    def download_volume_file(self, volume_path: str) -> Optional[bytes]:
        """Download a file from a Volume and return its bytes."""
        try:
            resp = self.workspace_client.files.download(volume_path)
            return resp.contents.read()
        except Exception as e:
            print(f"Error downloading {volume_path}: {e}")
            return None

    def download_volume_json(self, volume_path: str) -> Optional[Any]:
        """Download a JSON file from a Volume and parse it."""
        import json
        raw = self.download_volume_file(volume_path)
        if raw is None:
            return None
        return json.loads(raw)

    def download_volume_image(self, volume_path: str):
        """Download an image from a Volume and return a PIL Image."""
        from PIL import Image
        raw = self.download_volume_file(volume_path)
        if raw is None:
            return None
        return Image.open(io.BytesIO(raw))

    # ------------------------------------------------------------------ #
    #  Universal path helpers (auto-detect local vs /Volumes)             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_remote_path(path: str) -> bool:
        return path.startswith("/Volumes") or path.startswith("/Workspace")

    def read_json(self, path: str) -> Optional[Any]:
        """Read a JSON file from a local path or a /Volumes path."""
        import json
        if self._is_remote_path(path):
            return self.download_volume_json(path)
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return None

    def file_exists(self, path: str) -> bool:
        """Check if a file exists locally or on Volumes."""
        import os
        if self._is_remote_path(path):
            return self.download_volume_file(path) is not None
        return os.path.exists(path)

