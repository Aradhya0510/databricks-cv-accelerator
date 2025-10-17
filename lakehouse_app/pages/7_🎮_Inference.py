"""
Page 7: Inference Playground
Interactive model testing and inference
"""

import streamlit as st
import sys
from pathlib import Path
from PIL import Image
import io

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.state_manager import StateManager
from components.image_viewer import ImageViewer

# Initialize state
StateManager.initialize()

# Page config
st.title("🎮 Inference Playground")
st.markdown("Test your models with interactive inference")

# Check for selected model/endpoint
selected_model = StateManager.get("inference_model")
selected_endpoint = StateManager.get("inference_endpoint")

if not selected_model and not selected_endpoint:
    st.info("ℹ️ Select a model or endpoint to start testing")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📦 Browse Models", use_container_width=True):
            st.switch_page("pages/5_📦_Model_Registration.py")
    with col2:
        if st.button("🌐 Browse Endpoints", use_container_width=True):
            st.switch_page("pages/6_🌐_Deployment.py")
else:
    # Display selected model/endpoint
    if selected_endpoint:
        st.success(f"✅ Using endpoint: {selected_endpoint}")
        inference_type = "endpoint"
    else:
        st.success(f"✅ Using model: {selected_model.get('name', 'N/A')}")
        inference_type = "model"
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📷 Single Image", "📁 Batch Upload", "🔗 URL Input"])
    
    with tab1:
        st.markdown("### Single Image Inference")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
            help="Upload an image file for inference"
        )
        
        if uploaded_file:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            st.markdown("#### Original Image")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                ImageViewer.display_image(image)
            
            with col2:
                ImageViewer.display_image_info(image)
            
            st.markdown("---")
            
            # Inference controls
            st.markdown("#### Inference Settings")
            
            current_config = StateManager.get_current_config()
            task = current_config.get("model", {}).get("task_type", "detection") if current_config else "detection"
            
            col1, col2 = st.columns(2)
            
            with col1:
                if task == "detection":
                    confidence_threshold = st.slider(
                        "Confidence Threshold",
                        0.0, 1.0, 0.5, 0.05,
                        help="Minimum confidence for detections"
                    )
                elif task == "classification":
                    top_k = st.number_input(
                        "Top K Predictions",
                        1, 10, 5,
                        help="Number of top predictions to show"
                    )
            
            with col2:
                if task in ["semantic_segmentation", "instance_segmentation", "universal_segmentation"]:
                    mask_alpha = st.slider(
                        "Mask Transparency",
                        0.0, 1.0, 0.5, 0.05,
                        help="Transparency of segmentation masks"
                    )
            
            # Run inference
            if st.button("🚀 Run Inference", type="primary", use_container_width=True):
                with st.spinner("Running inference..."):
                    st.info("""
                    🔄 Inference process would:
                    1. Preprocess image
                    2. Run model/endpoint prediction
                    3. Postprocess results
                    4. Visualize predictions
                    """)
                    
                    # Mock predictions
                    st.success("✅ Inference complete!")
                    
                    st.markdown("#### Predictions")
                    
                    if task == "detection":
                        st.markdown("**Detected Objects:**")
                        st.info("Mock detection results would appear here with:")
                        st.markdown("- Bounding boxes overlaid on image")
                        st.markdown("- Class labels and confidence scores")
                        st.markdown("- Object count per class")
                        
                        # Mock results table
                        import pandas as pd
                        mock_detections = pd.DataFrame({
                            "Class": ["Person", "Car", "Dog"],
                            "Confidence": [0.95, 0.87, 0.82],
                            "BBox": ["[10, 20, 100, 200]", "[150, 30, 250, 180]", "[200, 150, 280, 220]"]
                        })
                        st.dataframe(mock_detections, use_container_width=True)
                    
                    elif task == "classification":
                        st.markdown("**Classification Results:**")
                        
                        # Mock classification results
                        import pandas as pd
                        mock_predictions = pd.DataFrame({
                            "Rank": [1, 2, 3, 4, 5],
                            "Class": ["Golden Retriever", "Labrador", "Poodle", "Beagle", "Bulldog"],
                            "Confidence": [0.85, 0.08, 0.04, 0.02, 0.01]
                        })
                        st.dataframe(mock_predictions, use_container_width=True)
                        
                        # Bar chart
                        import plotly.express as px
                        fig = px.bar(
                            mock_predictions,
                            x="Confidence",
                            y="Class",
                            orientation='h',
                            title="Top Predictions"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.markdown("**Segmentation Results:**")
                        st.info("Segmentation mask would be overlaid on the image")
                        st.markdown("- Color-coded classes")
                        st.markdown("- IoU scores")
                        st.markdown("- Per-class pixel counts")
                    
                    st.markdown("---")
                    
                    # Download results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            "📥 Download Annotated Image",
                            data=b"mock image data",
                            file_name="prediction.png",
                            mime="image/png"
                        )
                    
                    with col2:
                        st.download_button(
                            "📥 Download Results (JSON)",
                            data='{"predictions": "mock"}',
                            file_name="results.json",
                            mime="application/json"
                        )
        else:
            st.info("👆 Upload an image to start inference")
    
    with tab2:
        st.markdown("### Batch Image Upload")
        
        uploaded_files = st.file_uploader(
            "Upload multiple images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Upload multiple images for batch inference"
        )
        
        if uploaded_files:
            st.success(f"✅ Uploaded {len(uploaded_files)} image(s)")
            
            # Show thumbnails
            st.markdown("#### Uploaded Images")
            
            images = []
            for file in uploaded_files[:12]:  # Show first 12
                image = Image.open(file)
                images.append(image)
            
            ImageViewer.display_image_grid(images, columns=4)
            
            if len(uploaded_files) > 12:
                st.info(f"Showing first 12 of {len(uploaded_files)} images")
            
            st.markdown("---")
            
            # Batch inference
            col1, col2 = st.columns(2)
            
            with col1:
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=32,
                    value=8,
                    help="Number of images to process at once"
                )
            
            with col2:
                save_results = st.checkbox("Save Results to Storage", value=True)
            
            if st.button("🚀 Run Batch Inference", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Mock batch processing
                import time
                for i in range(len(uploaded_files)):
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {i+1}/{len(uploaded_files)} images...")
                    time.sleep(0.1)
                
                status_text.text("")
                st.success("✅ Batch inference complete!")
                
                st.info("💡 Results would be displayed in a grid with annotations")
    
    with tab3:
        st.markdown("### Image URL Input")
        
        image_url = st.text_input(
            "Image URL",
            placeholder="https://example.com/image.jpg",
            help="Enter a URL to an image"
        )
        
        if image_url:
            try:
                st.markdown("#### Image from URL")
                st.image(image_url, use_column_width=True)
                
                if st.button("🚀 Run Inference on URL", type="primary"):
                    st.info("Inference would run on the image from URL")
            
            except Exception as e:
                st.error(f"❌ Error loading image from URL: {str(e)}")
        else:
            st.info("👆 Enter an image URL to start inference")

# Sidebar
with st.sidebar:
    st.markdown("### 🎮 Inference Settings")
    
    if selected_endpoint:
        st.markdown(f"**Endpoint:** {selected_endpoint}")
    elif selected_model:
        st.markdown(f"**Model:** {selected_model.get('name', 'N/A')}")
    
    if st.button("🔄 Change Model", use_container_width=True):
        StateManager.set("inference_model", None)
        StateManager.set("inference_endpoint", None)
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    
    st.metric("Images Processed", "0")
    st.metric("Avg Inference Time", "N/A")
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    
    st.info("""
    - Upload high-quality images
    - Adjust confidence thresholds
    - Save results for later analysis
    - Test on diverse examples
    """)

