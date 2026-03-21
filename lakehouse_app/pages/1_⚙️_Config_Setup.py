"""
Page 1: Configuration Setup
Create and manage training configurations
"""

import streamlit as st
from pathlib import Path
import yaml
from datetime import datetime

from utils.config_generator import ConfigGenerator
from utils.state_manager import StateManager
from components.config_forms import ConfigFormBuilder
from components.theme import inject_theme, page_header, section_title, status_badge

inject_theme()
StateManager.initialize()

page_header("Configuration Setup", "Build and manage training configs for your CV models")

tab1, tab2, tab3 = st.tabs(["Create New Config", "Load Existing Config", "Recent Configs"])

with tab1:
    section_title("Step 1 — Task and Model")

    col1, col2 = st.columns(2)
    with col1:
        task = ConfigFormBuilder.task_selector()
        StateManager.set("selected_task", task)
    with col2:
        model_name, model_info = ConfigFormBuilder.model_selector(task)
        StateManager.set("selected_model", model_name)

    if model_info:
        st.info(f"Recommended batch size for {model_info['display']}: {ConfigGenerator.get_default_batch_size(model_name)}")

    # Step 2
    section_title("Step 2 — Data")
    current_config = StateManager.get_current_config()
    data_defaults = current_config.get("data", {}) if current_config else {}
    data_config = ConfigFormBuilder.data_config_form(task, data_defaults)

    # Step 3
    section_title("Step 3 — Training")
    training_defaults = current_config.get("training", {}) if current_config else {}
    training_config = ConfigFormBuilder.training_config_form(training_defaults)

    # Step 4
    section_title("Step 4 — MLflow Tracking")
    mlflow_defaults = {}
    if current_config:
        mlflow_defaults["experiment_name"] = current_config.get("mlflow", {}).get("experiment_name", "")
    mlflow_config = ConfigFormBuilder.mlflow_config_form(mlflow_defaults)

    # Step 5
    section_title("Step 5 — Output")
    output_defaults = current_config.get("output", {}) if current_config else {}
    output_config = ConfigFormBuilder.output_config_form(output_defaults)

    # Step 6
    section_title("Step 6 — Task-Specific Settings")
    model_specific_defaults = current_config.get("model", {}) if current_config else {}
    model_specific_config = ConfigFormBuilder.task_specific_config_form(task, model_specific_defaults)

    # Step 7
    section_title("Step 7 — Generate and Save")
    config_name = st.text_input(
        "Configuration Name",
        value=f"{task}_{model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Name for this configuration",
    )
    save_path = f"/tmp/configs/{config_name}.yaml"

    if st.button("Preview Configuration", type="secondary", use_container_width=True):
        try:
            config = ConfigGenerator.generate_config(
                task=task, model_name=model_name, data_config=data_config,
                training_config=training_config, mlflow_config=mlflow_config,
                output_config=output_config, model_specific_config=model_specific_config,
            )
            st.markdown(
                f'<div class="config-block">{ConfigGenerator.get_config_preview(config)}</div>',
                unsafe_allow_html=True,
            )
            is_valid, errors = ConfigGenerator.validate_config(config)
            if is_valid:
                st.success("Configuration is valid")
            else:
                st.error("Configuration has errors:")
                for error in errors:
                    st.error(f"  - {error}")
        except Exception as e:
            st.error(f"Error generating configuration: {e}")

    def _generate_and_validate():
        config = ConfigGenerator.generate_config(
            task=task, model_name=model_name, data_config=data_config,
            training_config=training_config, mlflow_config=mlflow_config,
            output_config=output_config, model_specific_config=model_specific_config,
        )
        is_valid, errors = ConfigGenerator.validate_config(config)
        return config, is_valid, errors

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Save Configuration", type="primary", use_container_width=True):
            try:
                config, is_valid, errors = _generate_and_validate()
                if not is_valid:
                    st.error("Cannot save invalid configuration:")
                    for error in errors:
                        st.error(f"  - {error}")
                else:
                    saved_path = ConfigGenerator.save_config(config, save_path)
                    StateManager.set_current_config(config, saved_path)
                    st.success(f"Configuration saved to: {saved_path}")
            except Exception as e:
                st.error(f"Error saving configuration: {e}")

    with col2:
        if st.button("Save & Set as Active", type="primary", use_container_width=True):
            try:
                config, is_valid, errors = _generate_and_validate()
                if not is_valid:
                    st.error("Cannot save invalid configuration:")
                    for error in errors:
                        st.error(f"  - {error}")
                else:
                    saved_path = ConfigGenerator.save_config(config, save_path)
                    StateManager.set_current_config(config, saved_path)
                    st.success("Configuration saved and set as active")
                    st.info(f"Path: {saved_path}")
            except Exception as e:
                st.error(f"Error saving configuration: {e}")

    with col3:
        if st.button("Reset Form", use_container_width=True):
            StateManager.set_current_config(None, None)
            st.rerun()

    try:
        preview_config = ConfigGenerator.generate_config(
            task=task, model_name=model_name, data_config=data_config,
            training_config=training_config, mlflow_config=mlflow_config,
            output_config=output_config, model_specific_config=model_specific_config,
        )
        st.download_button(
            label="Download Config YAML",
            data=yaml.dump(preview_config, default_flow_style=False, sort_keys=False),
            file_name=f"{config_name}.yaml",
            mime="application/x-yaml",
            use_container_width=True,
        )
    except Exception:
        pass

with tab2:
    section_title("Load Existing Configuration")

    config_path_input = st.text_input(
        "Configuration File Path", value="",
        placeholder="/Volumes/<catalog>/<schema>/<volume>/configs/my_config.yaml",
        help="Enter the path to an existing configuration YAML file",
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Load Configuration", type="primary", use_container_width=True):
            if not config_path_input:
                st.warning("Please enter a configuration file path")
            else:
                try:
                    config = ConfigGenerator.load_config(config_path_input)
                    is_valid, errors = ConfigGenerator.validate_config(config)
                    if is_valid:
                        StateManager.set_current_config(config, config_path_input)
                        st.success("Configuration loaded successfully")
                        st.info(f"Loaded from: {config_path_input}")
                        with st.expander("Configuration Preview", expanded=True):
                            st.code(ConfigGenerator.get_config_preview(config), language="yaml")
                    else:
                        st.error("Configuration has validation errors:")
                        for error in errors:
                            st.error(f"  - {error}")
                        StateManager.set_current_config(config, config_path_input)
                except FileNotFoundError:
                    st.error(f"File not found: {config_path_input}")
                except yaml.YAMLError as e:
                    st.error(f"Invalid YAML format: {e}")
                except Exception as e:
                    st.error(f"Error loading configuration: {e}")

    with col2:
        if st.button("Browse", use_container_width=True):
            st.info("File browser not yet implemented. Please enter the path manually.")

    current_config = StateManager.get_current_config()
    current_path = StateManager.get("config_path")
    if current_config and current_path:
        section_title("Currently Active Configuration")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Task", current_config.get("model", {}).get("task_type", "N/A"))
        with col2:
            st.metric("Model", current_config.get("model", {}).get("model_name", "N/A").split("/")[-1])
        with col3:
            st.metric("Batch Size", current_config.get("data", {}).get("batch_size", "N/A"))
        with st.expander("View Full Configuration"):
            st.code(ConfigGenerator.get_config_preview(current_config), language="yaml")

with tab3:
    section_title("Recent Configurations")
    recent_configs = StateManager.get("recent_configs", [])
    if not recent_configs:
        st.info("No recent configurations found. Create or load a configuration to get started.")
    else:
        st.markdown(f"Found {len(recent_configs)} recent configuration(s)")
        for idx, config_path in enumerate(recent_configs):
            with st.expander(f"{Path(config_path).name}", expanded=(idx == 0)):
                st.code(config_path)
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Load", key=f"load_{idx}", use_container_width=True):
                        try:
                            config = ConfigGenerator.load_config(config_path)
                            StateManager.set_current_config(config, config_path)
                            st.success(f"Loaded: {Path(config_path).name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error loading: {e}")
                with col2:
                    if st.button("Preview", key=f"preview_{idx}", use_container_width=True):
                        try:
                            config = ConfigGenerator.load_config(config_path)
                            st.code(ConfigGenerator.get_config_preview(config), language="yaml")
                        except Exception as e:
                            st.error(f"Error: {e}")
                with col3:
                    if st.button("Remove", key=f"remove_{idx}", use_container_width=True):
                        recent_configs.remove(config_path)
                        StateManager.set("recent_configs", recent_configs)
                        st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("### Current Status")
    current_config = StateManager.get_current_config()
    current_path = StateManager.get("config_path")
    if current_config and current_path:
        st.markdown(f'{status_badge("REGISTERED")}', unsafe_allow_html=True)
        st.markdown(f"**File:** {Path(current_path).name}")
        st.markdown(f"**Task:** {current_config.get('model', {}).get('task_type', 'N/A')}")
        st.markdown(f"**Model:** {current_config.get('model', {}).get('model_name', 'N/A').split('/')[-1]}")
        if st.button("Clear Active Config", use_container_width=True):
            StateManager.set_current_config(None, None)
            st.rerun()
    else:
        st.info("No active configuration")

    st.divider()
    st.markdown("### Quick Actions")
    if st.button("Go to Data EDA", use_container_width=True):
        st.switch_page("pages/2_📊_Data_EDA.py")
    if st.button("Go to Training", use_container_width=True):
        st.switch_page("pages/3_🚀_Training.py")
