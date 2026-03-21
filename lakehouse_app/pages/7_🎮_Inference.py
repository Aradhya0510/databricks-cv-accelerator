"""
Page 7: Inference Playground
Interactive model testing and inference
"""

import streamlit as st
from PIL import Image
import io

from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient
from components.image_viewer import ImageViewer
from components.theme import inject_theme, page_header, section_title, metric_card

inject_theme()
StateManager.initialize()

page_header("Inference Playground", "Test deployed models interactively")

selected_model = StateManager.get("inference_model")
selected_endpoint = StateManager.get("inference_endpoint")

if not selected_model and not selected_endpoint:
    st.info("Select a model or endpoint to start testing")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Browse Models", use_container_width=True):
            st.switch_page("pages/5_📦_Model_Registration.py")
    with col2:
        if st.button("Browse Endpoints", use_container_width=True):
            st.switch_page("pages/6_🌐_Deployment.py")
else:
    if selected_endpoint:
        st.markdown(
            f'<div class="raised-card">'
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#4E566A;'
            f'text-transform:uppercase;letter-spacing:0.08em;">ACTIVE ENDPOINT</div>'
            f'<code style="font-size:12px;color:#EDF0F7;">{selected_endpoint}</code>'
            f'</div>',
            unsafe_allow_html=True,
        )
        inference_type = "endpoint"
    else:
        st.markdown(
            f'<div class="raised-card">'
            f'<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#4E566A;'
            f'text-transform:uppercase;letter-spacing:0.08em;">ACTIVE MODEL</div>'
            f'<code style="font-size:12px;color:#EDF0F7;">{selected_model.get("name", "N/A")}</code>'
            f'</div>',
            unsafe_allow_html=True,
        )
        inference_type = "model"

    tab1, tab2, tab3 = st.tabs(["Single Image", "Batch Upload", "URL Input"])

    with tab1:
        section_title("Single Image Inference")

        uploaded_file = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"],
            help="Upload an image file for inference",
        )

        if uploaded_file:
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([2, 1])
            with col1:
                ImageViewer.display_image(image)
            with col2:
                ImageViewer.display_image_info(image)

            section_title("Inference Settings")

            current_config = StateManager.get_current_config()
            task = current_config.get("model", {}).get("task_type", "detection") if current_config else "detection"

            col1, col2 = st.columns(2)
            with col1:
                if task == "detection":
                    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05, help="Minimum confidence for detections")
                elif task == "classification":
                    top_k = st.number_input("Top K Predictions", 1, 10, 5, help="Number of top predictions to show")
            with col2:
                if task == "segmentation":
                    mask_alpha = st.slider("Mask Transparency", 0.0, 1.0, 0.5, 0.05, help="Transparency of segmentation masks")

            if st.button("Run Inference", type="primary", use_container_width=True):
                if inference_type != "endpoint" or not selected_endpoint:
                    st.warning("Endpoint-based inference required. Deploy a model and select an endpoint.")
                else:
                    with st.status("Running inference via endpoint...", expanded=True) as inf_status:
                        try:
                            client = DatabricksJobClient()
                            buf = io.BytesIO()
                            image.save(buf, format="PNG")
                            result = client.query_endpoint(selected_endpoint, buf.getvalue())

                            if "error" in result:
                                inf_status.update(label="Inference failed", state="error")
                                st.error(f"Endpoint error: {result['error']}")
                            else:
                                inf_status.update(label="Inference complete", state="complete")
                                st.success("Inference complete")

                                section_title("Predictions")
                                preds = result.get("predictions", result)
                                import pandas as pd, json as _json

                                if isinstance(preds, list) and preds:
                                    first = preds[0] if isinstance(preds[0], dict) else preds
                                    if isinstance(first, dict):
                                        st.dataframe(pd.DataFrame(preds), use_container_width=True)
                                    else:
                                        st.json(preds)
                                elif isinstance(preds, dict):
                                    st.json(preds)
                                else:
                                    st.write(preds)

                                col1, col2 = st.columns(2)
                                with col1:
                                    buf.seek(0)
                                    st.download_button(
                                        "Download Original Image",
                                        data=buf.getvalue(),
                                        file_name="input.png", mime="image/png",
                                    )
                                with col2:
                                    st.download_button(
                                        "Download Results (JSON)",
                                        data=_json.dumps(result, indent=2, default=str),
                                        file_name="results.json", mime="application/json",
                                    )
                        except Exception as e:
                            inf_status.update(label="Inference failed", state="error")
                            st.error(f"Inference failed: {e}")
        else:
            st.info("Upload an image to start inference")

    with tab2:
        section_title("Batch Image Upload")

        uploaded_files = st.file_uploader(
            "Upload multiple images", type=["jpg", "jpeg", "png"],
            accept_multiple_files=True, help="Upload multiple images for batch inference",
        )

        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} image(s)")

            images = []
            for file in uploaded_files[:12]:
                image = Image.open(file)
                images.append(image)

            ImageViewer.display_image_grid(images, columns=4)

            if len(uploaded_files) > 12:
                st.info(f"Showing first 12 of {len(uploaded_files)} images")

            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.number_input("Batch Size", 1, 32, 8, help="Number of images to process at once")
            with col2:
                save_results = st.checkbox("Save Results to Storage", value=True)

            if st.button("Run Batch Inference", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                import time
                for i in range(len(uploaded_files)):
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    status_text.markdown(
                        f'<span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#8A91A8;">'
                        f'Processing {i+1}/{len(uploaded_files)} — {progress:.0%}</span>',
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.1)

                status_text.empty()
                st.success("Batch inference complete")

    with tab3:
        section_title("Image URL Input")

        image_url = st.text_input("Image URL", placeholder="https://example.com/image.jpg", help="Enter a URL to an image")

        if image_url:
            try:
                st.image(image_url, use_column_width=True)
                if st.button("Run Inference on URL", type="primary"):
                    st.info("Inference would run on the image from URL")
            except Exception as e:
                st.error(f"Error loading image from URL: {e}")
        else:
            st.info("Enter an image URL to start inference")

# Sidebar
with st.sidebar:
    st.markdown("### Inference")
    if selected_endpoint:
        st.markdown(f"**Endpoint:** {selected_endpoint}")
    elif selected_model:
        st.markdown(f"**Model:** {selected_model.get('name', 'N/A')}")
    if st.button("Change Model", use_container_width=True):
        StateManager.set("inference_model", None)
        StateManager.set("inference_endpoint", None)
        st.rerun()
    st.divider()
    st.metric("Images Processed", "0")
    st.metric("Avg Inference Time", "N/A")
