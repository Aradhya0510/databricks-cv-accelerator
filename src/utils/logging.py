import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import mlflow
from lightning.pytorch.loggers import MLFlowLogger
from dataclasses import dataclass

@dataclass
class DatabricksLoggingConfig:
    """Simplified logging configuration for Databricks environment."""
    name: str = "cv_framework"
    level: int = logging.INFO
    
    # MLflow settings (primary for Databricks)
    experiment_name: Optional[str] = None
    tracking_uri: Optional[str] = None
    run_name: Optional[str] = None
    log_model: bool = True
    tags: Optional[Dict[str, Any]] = None
    
    # Optional file logging (for debugging)
    log_file: Optional[str] = None

class DatabricksLogger:
    """Streamlined logger for Databricks environment with managed MLflow."""
    
    def __init__(self, config: DatabricksLoggingConfig):
        """Initialize the Databricks logger.
        
        Args:
            config: Logging configuration optimized for Databricks
        """
        self.config = config
        self.python_logger = None
        self.mlflow_logger = None
        self._setup_python_logger()
        self._setup_mlflow_logger()
    
    def _setup_python_logger(self):
        """Setup Python logging for console output and optional file logging."""
        self.python_logger = logging.getLogger(self.config.name)
        self.python_logger.setLevel(self.config.level)

        # Remove any existing handlers
        self.python_logger.handlers = []

        # Console handler (always enabled for Databricks notebooks)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.python_logger.addHandler(console_handler)

        # Optional file handler for debugging
        if self.config.log_file:
            try:
                log_path = Path(self.config.log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(self.config.log_file)
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                ))
                self.python_logger.addHandler(file_handler)
            except Exception as e:
                self.python_logger.warning(f"Could not set up file logging: {e}")
    
    def _setup_mlflow_logger(self):
        """Setup MLflow logger for experiment tracking."""
        if self.config.experiment_name:
            self.mlflow_logger = MLFlowLogger(
                experiment_name=self.config.experiment_name,
                tracking_uri=self.config.tracking_uri,
                run_name=self.config.run_name,
                log_model=self.config.log_model,
                tags=self.config.tags
            )
    
    def info(self, message: str):
        """Log info message to console and optionally file."""
        if self.python_logger:
            self.python_logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        if self.python_logger:
            self.python_logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        if self.python_logger:
            self.python_logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        if self.python_logger:
            self.python_logger.debug(message)
    
    def get_mlflow_logger(self) -> Optional[MLFlowLogger]:
        """Get the MLflow logger for Lightning trainer."""
        return self.mlflow_logger
    
    def get_python_logger(self) -> Optional[logging.Logger]:
        """Get the Python logger for direct access."""
        return self.python_logger

def create_databricks_logger(
    name: str = "cv_framework",
    experiment_name: Optional[str] = None,
    tracking_uri: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> DatabricksLogger:
    """Create a Databricks-optimized logger.
    
    Args:
        name: Logger name
        experiment_name: MLflow experiment name (primary for Databricks)
        tracking_uri: MLflow tracking URI (usually auto-detected in Databricks)
        run_name: MLflow run name
        tags: MLflow tags
        log_file: Optional file path for debugging
        level: Logging level
        
    Returns:
        DatabricksLogger instance
    """
    config = DatabricksLoggingConfig(
        name=name,
        level=level,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name=run_name,
        tags=tags,
        log_file=log_file
    )
    
    return DatabricksLogger(config) 