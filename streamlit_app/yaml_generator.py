"""
YAML Generation and Validation
Converts configuration dictionaries to YAML format and validates them
"""

import yaml
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Add parent src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))


def generate_yaml(config_dict: Dict[str, Any]) -> str:
    """
    Generate YAML string from configuration dictionary.

    Args:
        config_dict: Configuration dictionary with model, data, training, mlflow, output sections

    Returns:
        YAML formatted string
    """
    # Clean up None values and empty dicts
    cleaned_config = _clean_config(config_dict)

    # Generate YAML with nice formatting
    yaml_str = yaml.dump(
        cleaned_config,
        default_flow_style=False,
        sort_keys=False,
        indent=2,
        allow_unicode=True
    )

    return yaml_str


def _clean_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove None values and empty dictionaries from config.

    Args:
        config: Configuration dictionary

    Returns:
        Cleaned configuration dictionary
    """
    if not isinstance(config, dict):
        return config

    cleaned = {}
    for key, value in config.items():
        if value is None:
            continue
        elif isinstance(value, dict):
            cleaned_dict = _clean_config(value)
            if cleaned_dict:  # Only add if not empty
                cleaned[key] = cleaned_dict
        elif isinstance(value, list):
            # Keep lists even if empty (e.g., image_size=[800, 800])
            cleaned[key] = value
        else:
            cleaned[key] = value

    return cleaned


def validate_config_dict(config_dict: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate configuration dictionary.

    Args:
        config_dict: Configuration dictionary to validate

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        # Basic structure validation
        required_sections = ["model", "data", "training", "mlflow", "output"]
        for section in required_sections:
            if section not in config_dict:
                return False, f"Missing required section: {section}"

        # Model section validation
        model = config_dict.get("model", {})
        required_model_fields = ["model_name", "task_type", "num_classes"]
        for field in required_model_fields:
            if field not in model:
                return False, f"Missing required model field: {field}"

        # Data section validation
        data = config_dict.get("data", {})
        required_data_fields = ["train_data_path", "val_data_path", "batch_size"]
        for field in required_data_fields:
            if field not in data:
                return False, f"Missing required data field: {field}"

        # Training section validation
        training = config_dict.get("training", {})
        required_training_fields = ["max_epochs", "learning_rate"]
        for field in required_training_fields:
            if field not in training:
                return False, f"Missing required training field: {field}"

        # Validate numeric ranges
        if model.get("num_classes", 0) <= 0:
            return False, "Number of classes must be greater than 0"

        if data.get("batch_size", 0) <= 0:
            return False, "Batch size must be greater than 0"

        if training.get("max_epochs", 0) <= 0:
            return False, "Max epochs must be greater than 0"

        if training.get("learning_rate", 0) <= 0:
            return False, "Learning rate must be greater than 0"

        # Validate paths (basic check)
        for path_key in ["train_data_path", "val_data_path"]:
            path = data.get(path_key, "")
            if not path or path.strip() == "":
                return False, f"{path_key} cannot be empty"

        # Try to import the actual validator if available
        try:
            from utils.config_validator import validate_config_for_simplified_mlflow

            # Convert to the format expected by the validator
            validated_config = validate_config_for_simplified_mlflow(config_dict)

            return True, "Configuration is valid and ready for deployment"

        except ImportError:
            # If validator not available, return success based on basic checks
            return True, "Configuration passed basic validation (full validator not available)"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    except Exception as e:
        return False, f"Unexpected validation error: {str(e)}"


def load_yaml_template(template_name: str) -> Dict[str, Any]:
    """
    Load a YAML template from the templates directory.

    Args:
        template_name: Name of the template file (without .yaml extension)

    Returns:
        Configuration dictionary
    """
    template_path = Path(__file__).parent / "templates" / f"{template_name}.yaml"

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    with open(template_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def save_yaml_to_file(yaml_str: str, file_path: str) -> bool:
    """
    Save YAML string to a file.

    Args:
        yaml_str: YAML formatted string
        file_path: Path to save the file

    Returns:
        True if successful, False otherwise
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w') as f:
            f.write(yaml_str)

        return True

    except Exception as e:
        print(f"Error saving YAML file: {e}")
        return False


def compare_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare two configurations and return differences.

    Args:
        config1: First configuration dictionary
        config2: Second configuration dictionary

    Returns:
        Dictionary of differences
    """
    differences = {}

    all_keys = set(list(config1.keys()) + list(config2.keys()))

    for key in all_keys:
        val1 = config1.get(key)
        val2 = config2.get(key)

        if val1 != val2:
            if isinstance(val1, dict) and isinstance(val2, dict):
                nested_diff = compare_configs(val1, val2)
                if nested_diff:
                    differences[key] = nested_diff
            else:
                differences[key] = {
                    "config1": val1,
                    "config2": val2
                }

    return differences
