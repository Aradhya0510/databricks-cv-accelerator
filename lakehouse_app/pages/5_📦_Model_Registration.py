"""
Page 5: Model Registration
Register trained models to Unity Catalog
"""

import streamlit as st
from datetime import datetime

from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient
from components.metrics_display import MetricsDisplay
from components.theme import inject_theme, page_header, section_title, metric_card, status_badge

inject_theme()
StateManager.initialize()

page_header("Model Registration", "Register models to Unity Catalog for versioning and deployment")

client = DatabricksJobClient()

tab1, tab2 = st.tabs(["Register Model", "Registered Models"])

with tab1:
    section_title("Step 1 — Select Training Run")

    config = StateManager.get_current_config() or {}
    default_exp = config.get("mlflow", {}).get("experiment_name", "")
    experiment_name = st.text_input(
        "MLflow Experiment", value=default_exp,
        help="Experiment name from your training config (auto-filled from active config)",
    )

    available_runs = []
    selected_run = None
    if experiment_name:
        with st.spinner("Loading runs..."):
            available_runs = client.get_mlflow_runs(experiment_name, max_results=20)

    if available_runs:
        run_labels = []
        for r in available_runs:
            status = r.get("status", "?")
            name = r.get("run_name", "unnamed")
            ts = r.get("start_time")
            ts_str = ts.strftime("%Y-%m-%d %H:%M") if ts else "—"
            run_labels.append(f"{name}  |  {status}  |  {ts_str}  |  {r['run_id'][:12]}…")

        choice = st.selectbox("Training Run", run_labels, index=0)
        idx = run_labels.index(choice)
        selected_run = available_runs[idx]

        metrics = selected_run.get("metrics", {})
        params = selected_run.get("params", {})
        model_uri_from_run = params.get("logged_model_uri", "")
        checkpoint_dir = params.get("checkpoint_dir", "")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Status", selected_run.get("status", "—"))
        with m2:
            primary = metrics.get("eval_map") or metrics.get("eval_accuracy")
            st.metric("Primary Metric", f"{primary:.4f}" if primary is not None else "—")
        with m3:
            st.metric("Eval Loss", f"{metrics['eval_loss']:.4f}" if "eval_loss" in metrics else "—")

        if model_uri_from_run:
            st.markdown(
                f'<div class="raised-card">'
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#4E566A;'
                f'text-transform:uppercase;letter-spacing:0.08em;">MODEL URI</div>'
                f'<code style="font-size:12px;color:#EDF0F7;">{model_uri_from_run}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )
        if checkpoint_dir:
            st.info(f"Checkpoint directory: `{checkpoint_dir}`")
    elif experiment_name:
        st.warning("No runs found in this experiment yet.")

    section_title("Step 2 — Model Registration Details")

    col1, col2, col3 = st.columns(3)
    with col1:
        catalog = st.text_input("Catalog", value="main", help="Unity Catalog name")
    with col2:
        schema = st.text_input("Schema", value="cv_models", help="Schema name")
    with col3:
        model_name = st.text_input(
            "Model Name",
            value=f"cv_model_{datetime.now().strftime('%Y%m%d')}",
            help="Name for the registered model",
        )

    full_model_name = f"{catalog}.{schema}.{model_name}"
    st.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:12px;color:#8A91A8;">'
        f'Full name: <code style="color:#EDF0F7;">{full_model_name}</code></div>',
        unsafe_allow_html=True,
    )

    section_title("Step 3 — Metadata")

    description = st.text_area(
        "Description", value="",
        placeholder="Enter a description for this model...",
        help="Describe the model, dataset, and any important details",
    )

    col1, col2 = st.columns(2)
    with col1:
        tags_input = st.text_input("Tags (comma-separated)", value="cv,pytorch,hf_trainer", help="Add tags for easy searching")
        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
    with col2:
        stage = st.selectbox("Initial Stage", options=["None", "Staging", "Production"], help="Set the initial stage")

    with st.expander("Advanced: manual Model URI / Run ID override"):
        model_uri_input = st.text_input("Model URI (overrides auto-discovery)", value="", help="Direct model URI")
        manual_run_id = st.text_input("MLflow Run ID (overrides selected run)", value="", help="Paste a run ID")

    section_title("Step 4 — Review and Register")

    eff_run_id = manual_run_id or (selected_run["run_id"] if selected_run else "")
    eff_model_uri = model_uri_input or (
        selected_run.get("params", {}).get("logged_model_uri", "") if selected_run else ""
    )

    st.markdown(
        f'<div class="surface-card">'
        f'<div class="detail-row"><span class="detail-label">Model</span>'
        f'<span class="detail-value"><code>{full_model_name}</code></span></div>'
        f'<div class="detail-row"><span class="detail-label">Source Run</span>'
        f'<span class="detail-value"><code>{eff_run_id[:16]}…</code></span></div>'
        f'<div class="detail-row"><span class="detail-label">Model URI</span>'
        f'<span class="detail-value"><code>{eff_model_uri or "will resolve from run"}</code></span></div>'
        f'<div class="detail-row"><span class="detail-label">Tags</span>'
        f'<span class="detail-value">{", ".join(tags)}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Register Model", type="primary", use_container_width=True):
            if not eff_run_id and not eff_model_uri:
                st.error("Select a training run or provide a Model URI / Run ID.")
            elif not full_model_name or "<" in full_model_name:
                st.error("Please provide a valid catalog.schema.model_name.")
            else:
                with st.status("Registering model to Unity Catalog...", expanded=True) as reg_status:
                    try:
                        import mlflow
                        mlflow.set_tracking_uri("databricks")

                        if eff_model_uri:
                            model_uri = eff_model_uri
                        else:
                            _client = mlflow.MlflowClient(tracking_uri="databricks")
                            _run = _client.get_run(eff_run_id)
                            stored = _run.data.params.get("logged_model_uri")
                            model_uri = stored if stored else f"runs:/{eff_run_id}/model"

                        mv = mlflow.register_model(model_uri, full_model_name)
                        version = mv.version

                        if description:
                            client.mlflow_client.update_registered_model(full_model_name, description=description)
                        for tag in tags:
                            client.mlflow_client.set_registered_model_tag(full_model_name, tag, "true")

                        StateManager.add_registered_model({
                            "name": full_model_name, "version": str(version),
                            "run_id": eff_run_id, "model_uri": model_uri,
                            "description": description, "tags": tags, "stage": stage,
                            "creation_timestamp": datetime.now().isoformat(),
                        })
                        reg_status.update(label="Registration complete", state="complete")
                        st.success(f"Registered **{full_model_name}** version {version}")
                        st.info("You can now deploy this model from the Deployment page")
                    except Exception as e:
                        reg_status.update(label="Registration failed", state="error")
                        st.error(f"Error registering model: {e}")
                        StateManager.add_registered_model({
                            "name": full_model_name, "version": "pending",
                            "run_id": eff_run_id, "description": description,
                            "tags": tags, "stage": stage,
                            "creation_timestamp": datetime.now().isoformat(),
                        })

    with col2:
        if st.button("Save as Draft", use_container_width=True):
            st.info("Draft saved locally")

with tab2:
    section_title("Registered Models")

    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Search Models", placeholder="Search by name, tag, or description...", label_visibility="collapsed")
    with col2:
        if st.button("Refresh List", use_container_width=True):
            st.rerun()

    try:
        mlflow_models = client.get_registered_models(max_results=50)
        local_models = StateManager.get("registered_models", [])
        all_models = mlflow_models if mlflow_models else local_models

        if not all_models:
            st.info("No registered models found")
        else:
            st.success(f"Found {len(all_models)} registered model(s)")

            if search_query:
                all_models = [
                    m for m in all_models
                    if search_query.lower() in m.get("name", "").lower()
                    or search_query.lower() in str(m.get("description", "")).lower()
                ]

            for model in all_models:
                with st.expander(f"{model.get('name', 'Unnamed Model')}"):
                    MetricsDisplay.display_model_card(model)

                    model_name_key = model.get("name", "").replace(".", "_").replace("-", "_")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        if st.button("Deploy", key=f"deploy_{model_name_key}", use_container_width=True):
                            StateManager.set("deployment_model", model)
                            st.switch_page("pages/6_🌐_Deployment.py")
                    with col2:
                        if st.button("Evaluate", key=f"eval_{model_name_key}", use_container_width=True):
                            st.switch_page("pages/4_📈_Evaluation.py")
                    with col3:
                        if st.button("Test", key=f"test_{model_name_key}", use_container_width=True):
                            StateManager.set("inference_model", model)
                            st.switch_page("pages/7_🎮_Inference.py")
                    with col4:
                        if st.button("Download", key=f"download_{model_name_key}", use_container_width=True):
                            st.info("Download functionality would export model artifacts")

    except Exception as e:
        st.error(f"Error loading models: {e}")
        local_models = StateManager.get("registered_models", [])
        if local_models:
            st.warning("Showing locally tracked models only")
            for model in local_models:
                with st.expander(f"{model.get('name', 'Unnamed Model')}"):
                    st.markdown(f"**Version:** {model.get('version', 'N/A')}")
                    st.markdown(f"**Created:** {model.get('creation_timestamp', 'N/A')}")

# Sidebar
with st.sidebar:
    st.markdown("### Model Registry")
    registered_models = StateManager.get("registered_models", [])
    st.metric("Registered Models", len(registered_models))
    st.divider()
    st.markdown("### Quick Actions")
    if st.button("Deploy Model", use_container_width=True):
        st.switch_page("pages/6_🌐_Deployment.py")
    if st.button("Test Model", use_container_width=True):
        st.switch_page("pages/7_🎮_Inference.py")
