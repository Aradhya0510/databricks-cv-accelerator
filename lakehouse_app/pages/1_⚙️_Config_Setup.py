"""
Page 1: Configuration Setup
Create and manage training configurations
"""

import streamlit as st
import sys
from pathlib import Path
import yaml
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config_generator import ConfigGenerator
from utils.state_manager import StateManager
from components.config_forms import ConfigFormBuilder

# Initialize state
StateManager.initialize()

# Page config
st.title("⚙️ Configuration Setup")
st.markdown("Create and manage training configurations for your computer vision models")

# Tabs for different workflows
tab1, tab2, tab3 = st.tabs(["🆕 Create New Config", "📂 Load Existing Config", "📋 Recent Configs"])

with tab1:
    st.markdown("### Create New Configuration")
    st.markdown("Fill out the form below to generate a training configuration")
    
    # Step 1: Task and Model Selection
    st.markdown("---")
    st.markdown("## Step 1: Select Task and Model")
    
    col1, col2 = st.columns(2)
    with col1:
        task = ConfigFormBuilder.task_selector()
        StateManager.set("selected_task", task)
    
    with col2:
        model_name, model_info = ConfigFormBuilder.model_selector(task)
        StateManager.set("selected_model", model_name)
    
    # Display recommended settings
    if model_info:
        st.info(f"💡 Recommended batch size for {model_info['display']}: {ConfigGenerator.get_default_batch_size(model_name)}")
    
    # Step 2: Data Configuration
    st.markdown("---")
    st.markdown("## Step 2: Configure Data")
    
    # Load defaults from state if available
    current_config = StateManager.get_current_config()
    data_defaults = current_config.get("data", {}) if current_config else {}
    
    data_config = ConfigFormBuilder.data_config_form(task, data_defaults)
    
    # Step 3: Training Configuration
    st.markdown("---")
    st.markdown("## Step 3: Configure Training")
    
    training_defaults = current_config.get("training", {}) if current_config else {}
    training_config = ConfigFormBuilder.training_config_form(training_defaults)
    
    # Step 4: MLflow Configuration
    st.markdown("---")
    st.markdown("## Step 4: Configure MLflow Tracking")
    
    mlflow_defaults = {}
    if current_config and "training" in current_config:
        mlflow_defaults["experiment_name"] = current_config["training"].get("experiment_name", "")
    
    mlflow_config = ConfigFormBuilder.mlflow_config_form(mlflow_defaults)
    
    # Step 5: Output Configuration
    st.markdown("---")
    st.markdown("## Step 5: Configure Output")
    
    output_defaults = current_config.get("output", {}) if current_config else {}
    output_config = ConfigFormBuilder.output_config_form(output_defaults)
    
    # Step 6: Task-Specific Configuration
    st.markdown("---")
    st.markdown("## Step 6: Task-Specific Settings")
    
    model_specific_defaults = current_config.get("model", {}) if current_config else {}
    model_specific_config = ConfigFormBuilder.task_specific_config_form(task, model_specific_defaults)
    
    # Generate Configuration
    st.markdown("---")
    st.markdown("## Step 7: Generate and Save Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        config_name = st.text_input(
            "Configuration Name",
            value=f"{task}_{model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            help="Name for this configuration"
        )
    
    with col2:
        save_path = st.text_input(
            "Save Path",
            value=f"/Volumes/<catalog>/<schema>/<volume>/configs/{config_name}.yaml",
            help="Path where the configuration will be saved"
        )
    
    # Preview button
    if st.button("👁️ Preview Configuration", type="secondary", use_container_width=True):
        try:
            config = ConfigGenerator.generate_config(
                task=task,
                model_name=model_name,
                data_config=data_config,
                training_config=training_config,
                mlflow_config=mlflow_config,
                output_config=output_config,
                model_specific_config=model_specific_config
            )
            
            st.markdown("### Configuration Preview")
            st.code(ConfigGenerator.get_config_preview(config), language="yaml")
            
            # Validate
            is_valid, errors = ConfigGenerator.validate_config(config)
            if is_valid:
                st.success("✅ Configuration is valid!")
            else:
                st.error("❌ Configuration has errors:")
                for error in errors:
                    st.error(f"  - {error}")
        
        except Exception as e:
            st.error(f"Error generating configuration: {str(e)}")
    
    # Save and apply buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Configuration", type="primary", use_container_width=True):
            try:
                # Generate config
                config = ConfigGenerator.generate_config(
                    task=task,
                    model_name=model_name,
                    data_config=data_config,
                    training_config=training_config,
                    mlflow_config=mlflow_config,
                    output_config=output_config,
                    model_specific_config=model_specific_config
                )
                
                # Validate
                is_valid, errors = ConfigGenerator.validate_config(config)
                if not is_valid:
                    st.error("❌ Cannot save invalid configuration. Please fix the errors:")
                    for error in errors:
                        st.error(f"  - {error}")
                else:
                    # Save to file
                    saved_path = ConfigGenerator.save_config(config, save_path)
                    st.success(f"✅ Configuration saved to: {saved_path}")
                    
                    # Update state
                    StateManager.set_current_config(config, saved_path)
                    
                    st.balloons()
            
            except Exception as e:
                st.error(f"❌ Error saving configuration: {str(e)}")
    
    with col2:
        if st.button("✅ Save & Set as Active", type="primary", use_container_width=True):
            try:
                # Generate config
                config = ConfigGenerator.generate_config(
                    task=task,
                    model_name=model_name,
                    data_config=data_config,
                    training_config=training_config,
                    mlflow_config=mlflow_config,
                    output_config=output_config,
                    model_specific_config=model_specific_config
                )
                
                # Validate
                is_valid, errors = ConfigGenerator.validate_config(config)
                if not is_valid:
                    st.error("❌ Cannot save invalid configuration. Please fix the errors:")
                    for error in errors:
                        st.error(f"  - {error}")
                else:
                    # Save to file
                    saved_path = ConfigGenerator.save_config(config, save_path)
                    
                    # Update state
                    StateManager.set_current_config(config, saved_path)
                    
                    st.success(f"✅ Configuration saved and set as active!")
                    st.info(f"📁 Path: {saved_path}")
                    st.info("💡 You can now proceed to Data EDA or Training pages")
                    
                    st.balloons()
            
            except Exception as e:
                st.error(f"❌ Error saving configuration: {str(e)}")
    
    with col3:
        if st.button("🔄 Reset Form", use_container_width=True):
            # Clear specific form-related state
            StateManager.set_current_config(None, None)
            st.rerun()

with tab2:
    st.markdown("### Load Existing Configuration")
    
    config_path_input = st.text_input(
        "Configuration File Path",
        value="",
        placeholder="/Volumes/<catalog>/<schema>/<volume>/configs/my_config.yaml",
        help="Enter the path to an existing configuration YAML file"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("📂 Load Configuration", type="primary", use_container_width=True):
            if not config_path_input:
                st.warning("Please enter a configuration file path")
            else:
                try:
                    config = ConfigGenerator.load_config(config_path_input)
                    
                    # Validate
                    is_valid, errors = ConfigGenerator.validate_config(config)
                    
                    if is_valid:
                        StateManager.set_current_config(config, config_path_input)
                        st.success(f"✅ Configuration loaded successfully!")
                        st.info(f"📁 Loaded from: {config_path_input}")
                        
                        # Display config preview
                        with st.expander("📄 Configuration Preview", expanded=True):
                            st.code(ConfigGenerator.get_config_preview(config), language="yaml")
                    else:
                        st.error("❌ Configuration has validation errors:")
                        for error in errors:
                            st.error(f"  - {error}")
                        st.info("The configuration was loaded but may not work correctly")
                        
                        # Still load it but warn user
                        StateManager.set_current_config(config, config_path_input)
                
                except FileNotFoundError:
                    st.error(f"❌ File not found: {config_path_input}")
                except yaml.YAMLError as e:
                    st.error(f"❌ Invalid YAML format: {str(e)}")
                except Exception as e:
                    st.error(f"❌ Error loading configuration: {str(e)}")
    
    with col2:
        # Browse button (would need file browser implementation)
        if st.button("🔍 Browse", use_container_width=True):
            st.info("File browser not yet implemented. Please enter the path manually.")
    
    # Show currently loaded config
    current_config = StateManager.get_current_config()
    current_path = StateManager.get("config_path")
    
    if current_config and current_path:
        st.markdown("---")
        st.markdown("### Currently Active Configuration")
        st.success(f"📁 **Path:** {current_path}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Task", current_config.get("model", {}).get("task_type", "N/A"))
        with col2:
            st.metric("Model", current_config.get("model", {}).get("model_name", "N/A").split("/")[-1])
        with col3:
            st.metric("Batch Size", current_config.get("data", {}).get("batch_size", "N/A"))
        
        with st.expander("📄 View Full Configuration"):
            st.code(ConfigGenerator.get_config_preview(current_config), language="yaml")

with tab3:
    st.markdown("### Recent Configurations")
    
    recent_configs = StateManager.get("recent_configs", [])
    
    if not recent_configs:
        st.info("No recent configurations found. Create or load a configuration to get started!")
    else:
        st.markdown(f"Found {len(recent_configs)} recent configuration(s)")
        
        for idx, config_path in enumerate(recent_configs):
            with st.expander(f"📄 {Path(config_path).name}", expanded=(idx == 0)):
                st.code(config_path)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📂 Load", key=f"load_{idx}", use_container_width=True):
                        try:
                            config = ConfigGenerator.load_config(config_path)
                            StateManager.set_current_config(config, config_path)
                            st.success(f"✅ Loaded: {Path(config_path).name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error loading: {str(e)}")
                
                with col2:
                    if st.button("👁️ Preview", key=f"preview_{idx}", use_container_width=True):
                        try:
                            config = ConfigGenerator.load_config(config_path)
                            st.code(ConfigGenerator.get_config_preview(config), language="yaml")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                
                with col3:
                    if st.button("🗑️ Remove", key=f"remove_{idx}", use_container_width=True):
                        recent_configs.remove(config_path)
                        StateManager.set("recent_configs", recent_configs)
                        st.success("Removed from recent list")
                        st.rerun()

# Sidebar: Current Status
with st.sidebar:
    st.markdown("### 📊 Current Status")
    
    current_config = StateManager.get_current_config()
    current_path = StateManager.get("config_path")
    
    if current_config and current_path:
        st.success("✅ Configuration Active")
        st.info(f"**File:** {Path(current_path).name}")
        st.markdown(f"**Task:** {current_config.get('model', {}).get('task_type', 'N/A')}")
        st.markdown(f"**Model:** {current_config.get('model', {}).get('model_name', 'N/A').split('/')[-1]}")
        
        if st.button("🗑️ Clear Active Config", use_container_width=True):
            StateManager.set_current_config(None, None)
            st.rerun()
    else:
        st.warning("⚠️ No active configuration")
        st.info("Create or load a configuration to get started")
    
    st.markdown("---")
    st.markdown("### 🚀 Quick Actions")
    
    if st.button("📊 Go to Data EDA", use_container_width=True):
        st.switch_page("pages/2_📊_Data_EDA.py")
    
    if st.button("🚀 Go to Training", use_container_width=True):
        st.switch_page("pages/3_🚀_Training.py")

