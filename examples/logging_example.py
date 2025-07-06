"""
Example demonstrating the Databricks-optimized logging system.

This example shows how to use the DatabricksLogger for both Python logging
and MLflow integration in a Databricks environment with managed MLflow.
"""

import os
from src.utils.logging import create_databricks_logger, DatabricksLogger, DatabricksLoggingConfig

def example_databricks_logging():
    """Example of Databricks-optimized logging setup."""
    print("=== Databricks Logging Example ===")
    
    # Create a logger optimized for Databricks environment
    logger = create_databricks_logger(
        name="databricks_example",
        experiment_name="test_experiment",  # Primary for Databricks
        run_name="test_run",
        tags={"framework": "lightning", "environment": "databricks"},
        # Optional file logging for debugging
        log_file="/tmp/databricks_example.log"
    )
    
    logger.info("Starting Databricks experiment")
    logger.info("This message goes to console and optionally file")
    logger.warning("This warning appears in console")
    logger.error("This error is logged for debugging")
    
    # Get MLflow logger for Lightning trainer
    mlflow_logger = logger.get_mlflow_logger()
    if mlflow_logger:
        print(f"MLflow logger ready for Lightning: {mlflow_logger}")
    
    print("Databricks logging complete")

def example_minimal_databricks_logging():
    """Example of minimal logging setup for Databricks."""
    print("\n=== Minimal Databricks Logging Example ===")
    
    # Minimal setup - just MLflow (most common in Databricks)
    logger = create_databricks_logger(
        name="minimal_example",
        experiment_name="minimal_experiment"
    )
    
    logger.info("Minimal setup - console logging + MLflow")
    logger.info("No file logging needed in Databricks")
    
    print("Minimal logging complete")

def example_debugging_logging():
    """Example of logging with debugging enabled."""
    print("\n=== Debugging Logging Example ===")
    
    # Setup with file logging for debugging
    logger = create_databricks_logger(
        name="debug_example",
        experiment_name="debug_experiment",
        log_file="/tmp/debug.log",  # For debugging
        level=logging.DEBUG  # More verbose
    )
    
    logger.debug("Debug message - only in file when debugging")
    logger.info("Info message - console and file")
    logger.warning("Warning - console and file")
    
    print("Debug logging complete. Check /tmp/debug.log")

def example_unified_trainer_integration():
    """Example of how to integrate with UnifiedTrainer in Databricks."""
    print("\n=== UnifiedTrainer Integration Example ===")
    
    # Create Databricks-optimized logger
    logger = create_databricks_logger(
        name="training_example",
        experiment_name="training_experiment",
        tags={"task": "detection", "model": "detr", "environment": "databricks"}
    )
    
    logger.info("Setting up training configuration")
    
    # Get MLflow logger for trainer
    mlflow_logger = logger.get_mlflow_logger()
    
    # Example of how to use with UnifiedTrainer
    # (This is just for demonstration - actual training would require model/data setup)
    logger.info("UnifiedTrainer uses MLflow logger for metrics tracking")
    logger.info("Python logger used for training progress and debugging")
    
    print("UnifiedTrainer integration example complete")

if __name__ == "__main__":
    import logging
    
    example_databricks_logging()
    example_minimal_databricks_logging()
    example_debugging_logging()
    example_unified_trainer_integration()
    
    print("\n=== All Examples Complete ===")
    print("The Databricks-optimized logging system provides:")
    print("1. MLflow-first approach (primary for Databricks)")
    print("2. Optional file logging for debugging")
    print("3. Console logging for immediate feedback")
    print("4. Simplified setup for Databricks environment")
    print("5. Clean, focused API without legacy code") 