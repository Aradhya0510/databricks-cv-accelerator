"""
Databricks Client for Jobs API and MLflow Integration
Handles job submission, monitoring, and MLflow interactions
"""

import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute
import mlflow
from mlflow.tracking import MlflowClient


class DatabricksJobClient:
    """Client for Databricks Jobs API and MLflow."""
    
    def __init__(self):
        """Initialize Databricks client."""
        self.workspace_client = WorkspaceClient()
        self.mlflow_client = MlflowClient()
    
    def create_training_job(
        self,
        job_name: str,
        config_path: str,
        src_path: str,
        cluster_config: Optional[Dict[str, Any]] = None,
        email_notifications: Optional[List[str]] = None,
    ) -> str:
        """
        Create a training job.
        
        Args:
            job_name: Name for the job
            config_path: Path to configuration YAML
            src_path: Path to src directory
            cluster_config: Optional cluster configuration
            email_notifications: Optional list of emails for notifications
            
        Returns:
            Job ID
        """
        # Default cluster configuration (GPU)
        if cluster_config is None:
            cluster_config = {
                "spark_version": "14.3.x-gpu-ml-scala2.12",
                "node_type_id": "g5.4xlarge",
                "num_workers": 0,  # Single node
                "runtime_engine": "STANDARD",
                "data_security_mode": "SINGLE_USER",
            }
        
        # Job task configuration
        task = jobs.Task(
            task_key="train_model",
            description="Train CV model",
            new_cluster=compute.ClusterSpec(**cluster_config),
            python_wheel_task=None,
            spark_python_task=jobs.SparkPythonTask(
                python_file=f"{src_path}/../jobs/model_training.py",
                parameters=[
                    "--config_path", config_path,
                    "--src_path", src_path
                ]
            ),
            libraries=[
                compute.Library(pypi=compute.PythonPyPiLibrary(package="torch>=2.0.0")),
                compute.Library(pypi=compute.PythonPyPiLibrary(package="torchvision>=0.15.0")),
                compute.Library(pypi=compute.PythonPyPiLibrary(package="transformers>=4.36.0")),
                compute.Library(pypi=compute.PythonPyPiLibrary(package="lightning>=2.1.0")),
                compute.Library(pypi=compute.PythonPyPiLibrary(package="mlflow>=2.10.0")),
            ],
            timeout_seconds=0,  # No timeout
        )
        
        # Email notifications
        notifications = None
        if email_notifications:
            notifications = jobs.JobEmailNotifications(
                on_success=email_notifications,
                on_failure=email_notifications,
            )
        
        # Create job
        created_job = self.workspace_client.jobs.create(
            name=job_name,
            tasks=[task],
            email_notifications=notifications,
            timeout_seconds=0,
            max_concurrent_runs=1,
        )
        
        return str(created_job.job_id)
    
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
        life_cycle_state = str(state.life_cycle_state) if state else "UNKNOWN"
        result_state = str(state.result_state) if state and state.result_state else "UNKNOWN"
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
    
    def get_mlflow_runs(
        self,
        experiment_name: str,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get MLflow runs for an experiment.
        
        Args:
            experiment_name: Name of the experiment
            max_results: Maximum number of runs to return
            
        Returns:
            List of run dictionaries
        """
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                return []
            
            runs = self.mlflow_client.search_runs(
                experiment_ids=[experiment.experiment_id],
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
                            "stage": version.current_stage,
                            "run_id": version.run_id,
                        }
                        for version in model.latest_versions
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
        Get serving endpoint status.
        
        Args:
            endpoint_name: Name of the endpoint
            
        Returns:
            Endpoint status dictionary
        """
        try:
            endpoint = self.workspace_client.serving_endpoints.get(endpoint_name)
            
            state = endpoint.state
            config_state = str(state.config_update) if state else "UNKNOWN"
            ready = state.ready if state else "UNKNOWN"
            
            return {
                "endpoint_name": endpoint_name,
                "state": config_state,
                "ready": ready,
                "endpoint_url": endpoint.url if hasattr(endpoint, 'url') else None,
            }
        except Exception as e:
            return {
                "endpoint_name": endpoint_name,
                "state": "NOT_FOUND",
                "error": str(e),
            }
    
    def list_clusters(self) -> List[Dict[str, Any]]:
        """
        List available clusters.
        
        Returns:
            List of cluster dictionaries
        """
        try:
            clusters = self.workspace_client.clusters.list()
            return [
                {
                    "cluster_id": cluster.cluster_id,
                    "cluster_name": cluster.cluster_name,
                    "state": str(cluster.state),
                    "node_type_id": cluster.node_type_id,
                    "num_workers": cluster.num_workers,
                }
                for cluster in clusters
            ]
        except Exception as e:
            print(f"Error listing clusters: {e}")
            return []

