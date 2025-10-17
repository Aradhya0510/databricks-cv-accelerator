"""
State Manager for Lakehouse App
Manages application state persistence and session state
"""

import streamlit as st
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime
import os


class StateManager:
    """Manage application state and persistence."""
    
    # Default state keys
    DEFAULT_STATE = {
        # Current configuration
        "current_config": None,
        "config_path": None,
        
        # Task and model selection
        "selected_task": "detection",
        "selected_model": None,
        
        # Training state
        "active_training_run": None,
        "training_history": [],
        
        # Registered models
        "registered_models": [],
        
        # Deployed endpoints
        "endpoints": [],
        
        # User preferences
        "default_catalog": "main",
        "default_schema": "cv_models",
        "default_volume": "cv_data",
        "workspace_email": None,
        
        # Recent activity
        "recent_configs": [],
        "recent_runs": [],
    }
    
    @classmethod
    def initialize(cls):
        """Initialize session state with default values."""
        for key, value in cls.DEFAULT_STATE.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get a value from session state.
        
        Args:
            key: State key
            default: Default value if key doesn't exist
            
        Returns:
            State value or default
        """
        return st.session_state.get(key, default)
    
    @classmethod
    def set(cls, key: str, value: Any):
        """
        Set a value in session state.
        
        Args:
            key: State key
            value: Value to set
        """
        st.session_state[key] = value
    
    @classmethod
    def update(cls, updates: Dict[str, Any]):
        """
        Update multiple state values at once.
        
        Args:
            updates: Dictionary of key-value pairs to update
        """
        for key, value in updates.items():
            st.session_state[key] = value
    
    @classmethod
    def get_current_config(cls) -> Optional[Dict[str, Any]]:
        """Get the current configuration."""
        return cls.get("current_config")
    
    @classmethod
    def set_current_config(cls, config: Dict[str, Any], config_path: Optional[str] = None):
        """
        Set the current configuration.
        
        Args:
            config: Configuration dictionary
            config_path: Optional path where config is saved
        """
        cls.set("current_config", config)
        if config_path:
            cls.set("config_path", config_path)
            cls.add_recent_config(config_path)
    
    @classmethod
    def add_recent_config(cls, config_path: str, max_recent: int = 10):
        """
        Add a config path to recent configs.
        
        Args:
            config_path: Path to configuration file
            max_recent: Maximum number of recent configs to keep
        """
        recent = cls.get("recent_configs", [])
        
        # Remove if already exists
        if config_path in recent:
            recent.remove(config_path)
        
        # Add to front
        recent.insert(0, config_path)
        
        # Limit size
        recent = recent[:max_recent]
        
        cls.set("recent_configs", recent)
    
    @classmethod
    def add_training_run(cls, run_info: Dict[str, Any]):
        """
        Add a training run to history.
        
        Args:
            run_info: Dictionary with run information
        """
        history = cls.get("training_history", [])
        
        # Add timestamp if not present
        if "timestamp" not in run_info:
            run_info["timestamp"] = datetime.now().isoformat()
        
        history.insert(0, run_info)
        
        # Keep last 50 runs
        history = history[:50]
        
        cls.set("training_history", history)
    
    @classmethod
    def set_active_training_run(cls, run_id: Optional[str]):
        """
        Set the active training run.
        
        Args:
            run_id: Run ID or None to clear
        """
        cls.set("active_training_run", run_id)
    
    @classmethod
    def get_active_training_run(cls) -> Optional[str]:
        """Get the active training run ID."""
        return cls.get("active_training_run")
    
    @classmethod
    def clear_active_training_run(cls):
        """Clear the active training run."""
        cls.set("active_training_run", None)
    
    @classmethod
    def add_registered_model(cls, model_info: Dict[str, Any]):
        """
        Add a registered model to the list.
        
        Args:
            model_info: Dictionary with model information
        """
        models = cls.get("registered_models", [])
        
        # Check if model already exists (by name)
        existing_idx = None
        for idx, model in enumerate(models):
            if model.get("name") == model_info.get("name"):
                existing_idx = idx
                break
        
        if existing_idx is not None:
            # Update existing model
            models[existing_idx] = model_info
        else:
            # Add new model
            models.insert(0, model_info)
        
        cls.set("registered_models", models)
    
    @classmethod
    def add_endpoint(cls, endpoint_info: Dict[str, Any]):
        """
        Add an endpoint to the list.
        
        Args:
            endpoint_info: Dictionary with endpoint information
        """
        endpoints = cls.get("endpoints", [])
        
        # Check if endpoint already exists (by name)
        existing_idx = None
        for idx, endpoint in enumerate(endpoints):
            if endpoint.get("endpoint_name") == endpoint_info.get("endpoint_name"):
                existing_idx = idx
                break
        
        if existing_idx is not None:
            # Update existing endpoint
            endpoints[existing_idx] = endpoint_info
        else:
            # Add new endpoint
            endpoints.insert(0, endpoint_info)
        
        cls.set("endpoints", endpoints)
    
    @classmethod
    def get_user_preferences(cls) -> Dict[str, Any]:
        """Get user preferences."""
        return {
            "default_catalog": cls.get("default_catalog", "main"),
            "default_schema": cls.get("default_schema", "cv_models"),
            "default_volume": cls.get("default_volume", "cv_data"),
            "workspace_email": cls.get("workspace_email"),
        }
    
    @classmethod
    def set_user_preferences(cls, preferences: Dict[str, Any]):
        """
        Set user preferences.
        
        Args:
            preferences: Dictionary with preference updates
        """
        cls.update(preferences)
    
    @classmethod
    def save_state_to_file(cls, file_path: str):
        """
        Save current state to a file.
        
        Args:
            file_path: Path to save state file
        """
        state_data = {
            key: st.session_state.get(key)
            for key in cls.DEFAULT_STATE.keys()
            if key in st.session_state
        }
        
        # Convert non-serializable objects
        def make_serializable(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj
        
        state_data = make_serializable(state_data)
        
        with open(file_path, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    @classmethod
    def load_state_from_file(cls, file_path: str):
        """
        Load state from a file.
        
        Args:
            file_path: Path to state file
        """
        if not os.path.exists(file_path):
            return
        
        with open(file_path, 'r') as f:
            state_data = json.load(f)
        
        cls.update(state_data)
    
    @classmethod
    def reset_state(cls):
        """Reset all state to defaults."""
        for key, value in cls.DEFAULT_STATE.items():
            st.session_state[key] = value
    
    @classmethod
    def export_config_history(cls) -> List[Dict[str, Any]]:
        """
        Export configuration history.
        
        Returns:
            List of config history entries
        """
        training_history = cls.get("training_history", [])
        recent_configs = cls.get("recent_configs", [])
        
        return {
            "training_history": training_history,
            "recent_configs": recent_configs,
            "exported_at": datetime.now().isoformat(),
        }
    
    @classmethod
    def get_default_paths(cls) -> Dict[str, str]:
        """
        Get default paths based on user preferences.
        
        Returns:
            Dictionary with default paths
        """
        prefs = cls.get_user_preferences()
        catalog = prefs.get("default_catalog", "main")
        schema = prefs.get("default_schema", "cv_models")
        volume = prefs.get("default_volume", "cv_data")
        email = prefs.get("workspace_email", "user@email.com")
        
        base_path = f"/Volumes/{catalog}/{schema}/{volume}"
        
        return {
            "data_path": f"{base_path}/data",
            "checkpoint_path": f"{base_path}/checkpoints",
            "volume_checkpoint_path": f"{base_path}/volume_checkpoints",
            "results_path": f"{base_path}/results",
            "configs_path": f"{base_path}/configs",
            "experiment_path": f"/Users/{email}/cv_experiments",
        }

