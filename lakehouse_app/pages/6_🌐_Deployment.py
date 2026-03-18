"""
Page 6 — Deployment
Deploy to Model Serving, manage endpoints, view status/URL/version.
"""

import streamlit as st
from datetime import datetime

from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient
from components.theme import inject_theme, page_header, metric_card, section_title, status_pill

inject_theme()
StateManager.initialize()

page_header("Deployment", "Deploy models and manage serving endpoints")

client = DatabricksJobClient()

tab_deploy, tab_endpoints, tab_batch = st.tabs(["Deploy Model", "Active Endpoints", "Batch Inference"])


# =========================== TAB 1 — Deploy ================================
with tab_deploy:
    deployment_model = StateManager.get("deployment_model")

    section_title("Select Model")

    if deployment_model:
        st.markdown(
            f'<div class="glass-card">'
            f'<strong style="color:#00D68F;">Selected:</strong> '
            f'<code>{deployment_model.get("name", "—")}</code> v{deployment_model.get("version", "1")}'
            f'</div>',
            unsafe_allow_html=True,
        )
        model_name = deployment_model.get("name", "")
        model_version = deployment_model.get("version", "1")
        if st.button("Choose different model"):
            StateManager.set("deployment_model", None)
            st.rerun()
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            catalog = st.text_input("Catalog", "main", key="dep_cat")
        with c2:
            schema = st.text_input("Schema", "cv_models", key="dep_schema")
        with c3:
            model_input = st.text_input("Model Name", "", key="dep_model")
        model_name = f"{catalog}.{schema}.{model_input}" if model_input else ""
        model_version = st.text_input("Version", "1", key="dep_ver")

    st.markdown("")
    section_title("Endpoint Configuration")

    endpoint_name = st.text_input(
        "Endpoint Name",
        value=f"{model_name.split('.')[-1] if model_name else 'model'}_endpoint",
        key="dep_ep",
    )

    c1, c2 = st.columns(2)
    with c1:
        workload_size = st.selectbox("Workload Size", ["Small", "Medium", "Large"], key="dep_wl")
    with c2:
        scale_to_zero = st.checkbox("Scale-to-Zero", True, key="dep_s2z")

    with st.expander("Workload Details"):
        st.markdown(
            "**Small** — dev/testing, lowest cost  \n"
            "**Medium** — moderate production traffic  \n"
            "**Large** — high-throughput production  \n\n"
            "Scale-to-zero shuts down when idle to save cost."
        )

    st.markdown("")
    section_title("Review")
    st.markdown(
        f'<div class="glass-card">'
        f'<div class="detail-row"><span class="detail-label">Model</span><span class="detail-value"><code>{model_name}</code></span></div>'
        f'<div class="detail-row"><span class="detail-label">Version</span><span class="detail-value">{model_version}</span></div>'
        f'<div class="detail-row"><span class="detail-label">Endpoint</span><span class="detail-value"><code>{endpoint_name}</code></span></div>'
        f'<div class="detail-row"><span class="detail-label">Size</span><span class="detail-value">{workload_size}</span></div>'
        f'<div class="detail-row"><span class="detail-label">Scale-to-Zero</span><span class="detail-value">{"Yes" if scale_to_zero else "No"}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        if st.button("Deploy to Endpoint", type="primary", use_container_width=True, key="do_deploy"):
            if not model_name:
                st.error("Please enter a model name.")
            elif not endpoint_name:
                st.error("Please enter an endpoint name.")
            else:
                with st.spinner("Deploying..."):
                    result = client.create_model_serving_endpoint(
                        endpoint_name=endpoint_name,
                        model_name=model_name,
                        model_version=model_version,
                        workload_size=workload_size,
                        scale_to_zero=scale_to_zero,
                    )
                if result.get("status") in ("created", "updated"):
                    st.success(f"Endpoint **{endpoint_name}** {result['status']}.")
                    StateManager.add_endpoint({
                        "endpoint_name": endpoint_name,
                        "model_name": model_name,
                        "model_version": model_version,
                        "workload_size": workload_size,
                        "scale_to_zero": scale_to_zero,
                        "created_at": datetime.now().isoformat(),
                    })
                else:
                    st.error(f"Deployment failed: {result.get('error', 'Unknown')}")
    with c2:
        if st.button("Save Config", use_container_width=True, key="dep_save"):
            st.info("Config saved")


# =========================== TAB 2 — Active Endpoints =====================
with tab_endpoints:
    section_title("Endpoints")

    c1, c2 = st.columns([4, 1])
    with c2:
        if st.button("Refresh", use_container_width=True, key="ep_refresh"):
            st.rerun()

    endpoints = StateManager.get("endpoints", [])

    if not endpoints:
        st.info("No tracked endpoints. Deploy a model to get started.")
    else:
        for ep in endpoints:
            ep_name = ep.get("endpoint_name", "")
            model = ep.get("model_name", "—")
            version = ep.get("model_version", "—")
            size = ep.get("workload_size", "—")
            created = ep.get("created_at", "—")[:10]

            # Fetch live status
            live = client.get_endpoint_status(ep_name)
            state_str = live.get("state", "UNKNOWN")
            ready_str = live.get("ready", "UNKNOWN")
            ep_url = live.get("endpoint_url", "")

            if state_str == "NOT_UPDATING" and ready_str == "READY":
                pill = status_pill("READY")
            elif state_str == "UPDATING" or ready_str != "READY":
                pill = status_pill("RUNNING")
            elif state_str == "NOT_FOUND":
                pill = status_pill("FAILED")
            else:
                pill = status_pill(state_str)

            st.markdown(
                f'<div class="endpoint-card">'
                f'<div class="endpoint-header">'
                f'<span class="endpoint-name">{ep_name}</span>'
                f'{pill}'
                f'</div>'
                f'<div class="detail-row"><span class="detail-label">Model</span><span class="detail-value"><code>{model}</code></span></div>'
                f'<div class="detail-row"><span class="detail-label">Version</span><span class="detail-value">{version}</span></div>'
                f'<div class="detail-row"><span class="detail-label">Workload</span><span class="detail-value">{size}</span></div>'
                f'<div class="detail-row"><span class="detail-label">Created</span><span class="detail-value">{created}</span></div>'
                + (f'<div class="detail-row"><span class="detail-label">URL</span><span class="detail-value"><a href="{ep_url}" style="color:#6C63FF;" target="_blank">{ep_url}</a></span></div>' if ep_url else "")
                + f'</div>',
                unsafe_allow_html=True,
            )

            # Served models detail
            served = live.get("served_models", [])
            if served:
                with st.expander(f"Served Entities — {ep_name}"):
                    for sm in served:
                        st.markdown(
                            f"**{sm.get('entity_name', '—')}** "
                            f"v{sm.get('entity_version', '—')}  |  "
                            f"Size: {sm.get('workload_size', '—')}"
                        )

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("Test", key=f"t_{ep_name}", use_container_width=True):
                    StateManager.set("inference_endpoint", ep_name)
                    st.switch_page("pages/7_🎮_Inference.py")
            with c2:
                if st.button("Monitor", key=f"m_{ep_name}", use_container_width=True):
                    st.switch_page("pages/9_📡_Monitoring.py")
            with c3:
                if st.button("Smoke Test", key=f"s_{ep_name}", use_container_width=True):
                    st.code(
                        f'python -c "from src.serving.deployment import test_endpoint; '
                        f"test_endpoint('{ep_name}', test_image_path='<path>')\"",
                        language="bash",
                    )
            with c4:
                if st.button("Delete", key=f"d_{ep_name}", use_container_width=True):
                    st.warning("Confirm delete? (not yet implemented)")
            st.markdown("")


# =========================== TAB 3 — Batch Inference ======================
with tab_batch:
    section_title("Batch Inference")
    st.info("Process large datasets offline via a Databricks job.")

    c1, c2 = st.columns(2)
    with c1:
        inf_type = st.radio("Inference source", ["Endpoint", "Checkpoint"], key="bi_src")
    with c2:
        if inf_type == "Endpoint":
            ep_names = [e["endpoint_name"] for e in StateManager.get("endpoints", [])]
            bi_ep = st.selectbox("Endpoint", ep_names, key="bi_ep") if ep_names else None
        else:
            bi_ckpt = st.text_input("Checkpoint Path", "/Volumes/.../checkpoints/model", key="bi_ckpt")

    input_path = st.text_input("Input Path", "/Volumes/.../inference/input", key="bi_in")
    output_path = st.text_input("Output Path", "/Volumes/.../inference/output", key="bi_out")
    output_format = st.selectbox("Format", ["Delta Table", "Parquet", "JSON", "CSV"], key="bi_fmt")

    if st.button("Launch Batch Inference", type="primary", key="bi_go"):
        st.success("Batch job would be submitted here.")


# =========================== Sidebar ========================================
with st.sidebar:
    st.markdown("### Deployment")
    n = len(StateManager.get("endpoints", []))
    st.metric("Endpoints", n)
    st.divider()
    st.info("Test endpoints with the Inference page. Monitor health via Monitoring.")
