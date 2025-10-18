"""
Databricks Computer Vision Training Pipeline - Lakehouse App
Main entry point and navigation
"""

import streamlit as st
from pathlib import Path
import sys

# Note: The lakehouse_app is self-contained and doesn't need parent directory imports
# All required modules are within the lakehouse_app directory structure

# Configure page
st.set_page_config(
    page_title="CV Training Pipeline",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF3621;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .feature-card {
        background-color: #F0F2F6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main landing page."""
    
    # Header
    st.markdown('<div class="main-header">🎯 Computer Vision Training Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">End-to-End ML Lifecycle for Computer Vision Models on Databricks</div>', unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("---")
    st.markdown("### 👋 Welcome to the CV Training Pipeline!")
    st.markdown("""
    This application provides a **no-code/low-code interface** to train, evaluate, and deploy 
    computer vision models on Databricks. Navigate through the pages using the sidebar to:
    """)
    
    # Feature cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 🛠️ **Configuration Setup**")
        st.markdown("""
        - Select your CV task (Detection, Classification, Segmentation)
        - Choose from pre-trained models
        - Configure training parameters with intuitive forms
        - Auto-generate YAML configurations
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 **Data Exploration**")
        st.markdown("""
        - Visualize your dataset statistics
        - Preview image samples and annotations
        - Validate data quality
        - Explore class distributions
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 🚀 **Model Training**")
        st.markdown("""
        - Launch training jobs with one click
        - Monitor training progress in real-time
        - Track metrics and visualize learning curves
        - Hyperparameter tuning support
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 📈 **Model Evaluation**")
        st.markdown("""
        - Comprehensive performance metrics
        - Visual prediction analysis
        - Error analysis and confusion matrices
        - Compare multiple model runs
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 📦 **Model Registration**")
        st.markdown("""
        - Register models to Unity Catalog
        - Automatic model versioning
        - Generate model cards and documentation
        - Track model lineage
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 🌐 **Model Deployment**")
        st.markdown("""
        - Deploy to Databricks Model Serving
        - Configure endpoints and scaling
        - Batch inference configuration
        - Test deployed models
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 🎮 **Inference Playground**")
        st.markdown("""
        - Interactive model testing
        - Upload images for prediction
        - Real-time visualization of results
        - Export predictions and annotated images
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown("#### 📜 **History & Management**")
        st.markdown("""
        - Track all pipeline runs
        - Manage experiments and jobs
        - Resource usage analytics
        - Export and share configurations
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick start
    st.markdown("---")
    st.markdown("### 🚀 Quick Start")
    
    st.info("""
    **New to the pipeline?** Start with the **⚙️ Config Setup** page to configure your first training job!
    
    **Have an existing config?** Jump to **📊 Data EDA** to explore your dataset or **🚀 Training** to start training immediately.
    """)
    
    # Supported tasks
    st.markdown("---")
    st.markdown("### 📋 Supported Tasks")
    
    task_col1, task_col2, task_col3, task_col4, task_col5 = st.columns(5)
    
    with task_col1:
        st.markdown("**🔍 Detection**")
        st.markdown("DETR, YOLOS")
    
    with task_col2:
        st.markdown("**🏷️ Classification**")
        st.markdown("ResNet, ViT")
    
    with task_col3:
        st.markdown("**🎨 Semantic Seg.**")
        st.markdown("SegFormer, MiT")
    
    with task_col4:
        st.markdown("**🖼️ Instance Seg.**")
        st.markdown("Mask2Former")
    
    with task_col5:
        st.markdown("**🌐 Universal Seg.**")
        st.markdown("Mask2Former")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        Built on Databricks Lakehouse Platform | Powered by PyTorch Lightning & MLflow
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

