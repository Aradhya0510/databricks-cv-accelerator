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

from components.theme import inject_theme, metric_card, status_pill
from utils.state_manager import StateManager

inject_theme()
StateManager.initialize()


def _quick_stat(label, value, icon=""):
    return (
        f'<div class="glass-card" style="text-align:center;padding:1.2rem;">'
        f'<div style="font-size:1.3rem;margin-bottom:0.3rem;">{icon}</div>'
        f'<div style="font-size:1.6rem;font-weight:700;color:#fff;">{value}</div>'
        f'<div style="font-size:0.78rem;color:#8B949E;text-transform:uppercase;'
        f'letter-spacing:0.04em;margin-top:0.2rem;">{label}</div>'
        f'</div>'
    )


def main():
    # Hero banner
    st.markdown(
        '<div class="hero-banner">'
        "<h1>Computer Vision Pipeline</h1>"
        "<p>End-to-end training, evaluation, and deployment for object detection "
        "and image classification on Databricks</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Quick-stats row
    config = StateManager.get_current_config()
    training_history = StateManager.get("training_history", [])
    endpoints = StateManager.get("endpoints", [])
    active_run = StateManager.get_active_training_run()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        task = config.get("model", {}).get("task_type", "—").replace("_", " ").title() if config else "—"
        st.markdown(_quick_stat("Active Task", task, ""), unsafe_allow_html=True)
    with c2:
        model = config.get("model", {}).get("model_name", "—").split("/")[-1] if config else "—"
        st.markdown(_quick_stat("Model", model, ""), unsafe_allow_html=True)
    with c3:
        st.markdown(
            _quick_stat("Training Runs", str(len(training_history)), ""),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            _quick_stat("Endpoints", str(len(endpoints)), ""),
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Live training indicator
    if active_run:
        st.markdown(
            f'<div class="glass-card" style="display:flex;align-items:center;gap:1rem;">'
            f'<div style="width:10px;height:10px;border-radius:50%;background:#FFAA00;'
            f'animation:pulse 1.5s infinite;"></div>'
            f'<div><strong style="color:#E6EDF3;">Training in progress</strong>'
            f'<span style="color:#8B949E;margin-left:0.5rem;">Run ID: {str(active_run)[:12]}...</span></div>'
            f'<style>@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}</style>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    # Navigation cards
    nav_items = [
        ("", "Config Setup", "Build YAML configs with an interactive form. Select models, data paths, and hyperparameters.", "pages/1_⚙️_Config_Setup.py"),
        ("", "Data Explorer", "Visualize images with annotations overlaid, analyze class distributions, and spot data issues.", "pages/2_📊_Data_EDA.py"),
        ("", "Training", "Launch jobs, monitor loss curves and epoch progress live from MLflow.", "pages/3_🚀_Training.py"),
        ("", "Evaluation", "Inspect metrics, view predictions on images, and compare model runs.", "pages/4_📈_Evaluation.py"),
        ("", "Registration", "Register models to Unity Catalog with versioning and lineage.", "pages/5_📦_Model_Registration.py"),
        ("", "Deployment", "Deploy to Model Serving, view endpoint URLs and health status.", "pages/6_🌐_Deployment.py"),
        ("", "Inference", "Test deployed models interactively with image uploads.", "pages/7_🎮_Inference.py"),
        ("", "Monitoring", "Track endpoint health, request metrics, and prediction drift.", "pages/9_📡_Monitoring.py"),
    ]

    cols = st.columns(4)
    for idx, (icon, title, desc, page) in enumerate(nav_items):
        with cols[idx % 4]:
            st.markdown(
                f'<div class="nav-card">'
                f'<div class="nav-icon">{icon}</div>'
                f"<h3>{title}</h3>"
                f"<p>{desc}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if st.button(f"Open {title}", key=f"nav_{idx}", use_container_width=True):
                st.switch_page(page)

    # Supported tasks
    st.markdown("")
    st.markdown(
        '<div class="section-title">Supported Tasks</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            '<div class="glass-card">'
            '<strong style="color:#E6EDF3;">Object Detection</strong>'
            '<p style="color:#8B949E;font-size:0.88rem;margin:0.4rem 0 0 0;">'
            "DETR, YOLOS &mdash; COCO-format annotations, mAP evaluation</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="glass-card">'
            '<strong style="color:#E6EDF3;">Image Classification</strong>'
            '<p style="color:#8B949E;font-size:0.88rem;margin:0.4rem 0 0 0;">'
            "ViT, ResNet, any AutoModelForImageClassification &mdash; ImageFolder layout</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    # Footer
    st.markdown(
        '<div style="text-align:center;color:#8B949E;padding:2rem 0 1rem 0;font-size:0.82rem;">'
        "Built on Databricks Lakehouse &bull; HuggingFace Trainer &bull; MLflow 3"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
