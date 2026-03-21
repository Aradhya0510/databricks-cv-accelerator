"""
Databricks Computer Vision Pipeline — Lakehouse App
Main entry point and dashboard
"""

import streamlit as st

st.set_page_config(
    page_title="CV Pipeline",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.theme import inject_theme, metric_card, status_badge, section_title
from utils.state_manager import StateManager

inject_theme()
StateManager.initialize()


def main():
    st.markdown(
        '<div class="hero-banner">'
        "<h1>Computer Vision Pipeline</h1>"
        "<p>End-to-end training, evaluation, and deployment for object detection, "
        "image classification, and segmentation on Databricks</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    config = StateManager.get_current_config()
    training_history = StateManager.get("training_history", [])
    endpoints = StateManager.get("endpoints", [])
    active_run = StateManager.get_active_training_run()

    # Three metric cards — always at the top
    c1, c2, c3 = st.columns(3)
    with c1:
        task = (
            config.get("model", {}).get("task_type", "—").replace("_", " ").title()
            if config
            else "—"
        )
        metric_card("Active Task", task)
    with c2:
        model = (
            config.get("model", {}).get("model_name", "—").split("/")[-1]
            if config
            else "—"
        )
        metric_card("Model", model)
    with c3:
        metric_card("Training Runs", str(len(training_history)))

    st.markdown("")

    # Live training indicator
    if active_run:
        st.markdown(
            f'<div class="surface-card" style="display:flex;align-items:center;gap:12px;">'
            f'<div style="width:8px;height:8px;border-radius:50%;background:#F4A742;'
            f'animation:pulse 1.5s infinite;"></div>'
            f'<div>'
            f'<span style="font-family:Syne,sans-serif;font-weight:600;color:#EDF0F7;">Training in progress</span>'
            f'<span style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#4E566A;'
            f'margin-left:8px;">Run {str(active_run)[:12]}…</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    # Navigation cards
    nav_items = [
        ("Config Setup", "Build YAML configs with an interactive form. Select models, data paths, and hyperparameters.", "pages/1_⚙️_Config_Setup.py"),
        ("Data Explorer", "Visualize images with annotations overlaid, analyze class distributions, and spot data issues.", "pages/2_📊_Data_EDA.py"),
        ("Training", "Launch jobs, monitor loss curves and epoch progress live from MLflow.", "pages/3_🚀_Training.py"),
        ("Evaluation", "Inspect metrics, view predictions on images, and compare model runs.", "pages/4_📈_Evaluation.py"),
        ("Registration", "Register models to Unity Catalog with versioning and lineage.", "pages/5_📦_Model_Registration.py"),
        ("Deployment", "Deploy to Model Serving, view endpoint URLs and health status.", "pages/6_🌐_Deployment.py"),
        ("Inference", "Test deployed models interactively with image uploads.", "pages/7_🎮_Inference.py"),
        ("Monitoring", "Track endpoint health, request metrics, and prediction drift.", "pages/9_📡_Monitoring.py"),
    ]

    cols = st.columns(4)
    for idx, (title, desc, page) in enumerate(nav_items):
        with cols[idx % 4]:
            st.markdown(
                f'<div class="nav-card">'
                f"<h3>{title}</h3>"
                f"<p>{desc}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button(f"Open {title}", key=f"nav_{idx}", use_container_width=True):
                st.switch_page(page)

    # Supported tasks
    st.markdown("")
    section_title("Supported Tasks")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="raised-card">'
            f'<strong style="color:#EDF0F7;font-family:Syne,sans-serif;font-size:14px;">Object Detection</strong>'
            f'<p style="color:#8A91A8;font-size:12px;font-family:Figtree,sans-serif;margin:6px 0 0 0;line-height:1.6;">'
            "DETR, YOLOS, Conditional DETR, RT-DETR, DETA — COCO-format, mAP evaluation</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="raised-card">'
            f'<strong style="color:#EDF0F7;font-family:Syne,sans-serif;font-size:14px;">Image Classification</strong>'
            f'<p style="color:#8A91A8;font-size:12px;font-family:Figtree,sans-serif;margin:6px 0 0 0;line-height:1.6;">'
            "ViT, ResNet, ConvNeXT, Swin — ImageFolder layout, accuracy/F1 metrics</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="raised-card">'
            f'<strong style="color:#EDF0F7;font-family:Syne,sans-serif;font-size:14px;">Image Segmentation</strong>'
            f'<p style="color:#8A91A8;font-size:12px;font-family:Figtree,sans-serif;margin:6px 0 0 0;line-height:1.6;">'
            "SegFormer, Mask2Former — COCO instances/panoptic or ADE20K, mIoU metrics</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown(
        f'<div style="text-align:center;color:#4E566A;padding:32px 0 16px 0;'
        f'font-family:IBM Plex Mono,monospace;font-size:11px;letter-spacing:0.05em;">'
        "BUILT ON DATABRICKS LAKEHOUSE &bull; HUGGINGFACE TRAINER &bull; MLFLOW 3"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
