"""
Page 3 — Training
Launch jobs, monitor loss/epoch in real-time, and link to MLflow UI.
"""

import streamlit as st
from pathlib import Path
from datetime import datetime
import time
import yaml

from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient
from utils.config_generator import ConfigGenerator
from components.theme import inject_theme, page_header, metric_card, section_title, status_pill
from components.visualizations import VisualizationHelper

inject_theme()
StateManager.initialize()

page_header("Training", "Launch jobs, track loss & metrics in real-time")

config = StateManager.get_current_config()
config_path = StateManager.get("config_path")

if not config:
    st.warning("No active config — set one up first.")
    if st.button("Open Config Setup"):
        st.switch_page("pages/1_⚙️_Config_Setup.py")
    st.stop()

task = config.get("model", {}).get("task_type", "N/A")
model_name = config.get("model", {}).get("model_name", "N/A")
epochs = config.get("training", {}).get("max_epochs", "N/A")
batch_size = config.get("data", {}).get("batch_size", "N/A")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_launch, tab_monitor, tab_history = st.tabs(["Launch", "Live Dashboard", "History"])


# =========================== TAB 1 — Launch ================================
with tab_launch:
    section_title("Active Configuration")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Task", task.replace("_", " ").title())
    with c2:
        metric_card("Model", model_name.split("/")[-1])
    with c3:
        metric_card("Epochs", str(epochs))
    with c4:
        metric_card("Batch Size", str(batch_size))

    with st.expander("View YAML"):
        st.code(ConfigGenerator.get_config_preview(config), language="yaml")

    st.markdown("")
    section_title("Job Settings")

    c1, c2 = st.columns(2)
    with c1:
        job_name = st.text_input("Job Name", value=f"cv_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    with c2:
        project_path = st.text_input(
            "Project Path",
            value="/Workspace/Users/<username>/databricks-cv-accelerator",
            help="Workspace path containing jobs/ and src/",
        )

    section_title("Compute")
    c1, c2 = st.columns(2)
    with c1:
        node_type = st.selectbox(
            "Node Type",
            ["g5.4xlarge (1 GPU)", "g5.8xlarge (1 GPU)", "g5.12xlarge (4 GPU)", "g5.24xlarge (4 GPU)"],
        )
        node_type_id = node_type.split(" ")[0]
    with c2:
        num_workers = st.number_input("Workers (0 = single-node)", 0, 10, 0)

    with st.expander("Email Notifications"):
        emails_raw = st.text_input("Addresses (comma-separated)", "")
        emails = [e.strip() for e in emails_raw.split(",") if e.strip()] if emails_raw else []

    st.markdown("")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        if st.button("Launch Training", type="primary", use_container_width=True):
            if not config_path:
                st.error("Save your configuration first.")
            elif "<username>" in project_path:
                st.error("Update the project path with your actual username.")
            else:
                with st.spinner("Creating job and uploading config..."):
                    client = DatabricksJobClient()
                    cfg_filename = Path(config_path).name if config_path else f"{job_name}.yaml"
                    data_cfg = config.get("data", {})
                    tp = data_cfg.get("train_data_path", "")
                    if tp.startswith("/Volumes"):
                        parts = tp.strip("/").split("/")
                        vol_base = "/".join(parts[:4])
                        vol_cfg_dir = f"/{vol_base}/configs"
                    else:
                        vol_cfg_dir = "/tmp/configs"
                    remote_cfg = f"{vol_cfg_dir}/{cfg_filename}"
                    cfg_bytes = yaml.dump(config, default_flow_style=False, sort_keys=False).encode()
                    import io as _io

                    if remote_cfg.startswith("/Volumes"):
                        client.workspace_client.files.upload(remote_cfg, _io.BytesIO(cfg_bytes), overwrite=True)
                    else:
                        Path(vol_cfg_dir).mkdir(parents=True, exist_ok=True)
                        with open(remote_cfg, "wb") as f:
                            f.write(cfg_bytes)

                    cluster_config = {
                        "spark_version": "17.3.x-gpu-ml-scala2.13",
                        "node_type_id": node_type_id,
                        "num_workers": num_workers,
                        "data_security_mode": "SINGLE_USER",
                    }
                    job_id = client.create_training_job(
                        job_name=job_name,
                        config_path=remote_cfg,
                        project_path=project_path,
                        cluster_config=cluster_config,
                        email_notifications=emails or None,
                    )
                    run_id = client.run_job(job_id)
                    StateManager.set_active_training_run(run_id)
                    StateManager.add_training_run({
                        "run_id": run_id, "job_id": job_id, "job_name": job_name,
                        "config_path": config_path, "task": task,
                        "model": model_name, "timestamp": datetime.now().isoformat(),
                        "status": "RUNNING",
                    })

                st.success(f"Job launched — Run ID `{run_id}`")
                st.info("Switch to the **Live Dashboard** tab to monitor progress.")
    with c2:
        if st.button("Save Config First", use_container_width=True):
            st.switch_page("pages/1_⚙️_Config_Setup.py")
    with c3:
        if st.button("View Data", use_container_width=True):
            st.switch_page("pages/2_📊_Data_EDA.py")


# =========================== TAB 2 — Live Dashboard ========================
with tab_monitor:
    active_run_id = StateManager.get_active_training_run()

    if not active_run_id:
        st.info("No active training run. Launch one from the **Launch** tab.")
    else:
        client = DatabricksJobClient()
        status = client.get_job_status(active_run_id)
        state = status.get("life_cycle_state", "UNKNOWN")
        result = status.get("result_state", "UNKNOWN")

        # Status header
        st.markdown(
            f'<div class="glass-card" style="display:flex;align-items:center;justify-content:space-between;">'
            f'<div><strong style="color:#E6EDF3;">Run {str(active_run_id)[:12]}...</strong></div>'
            f'<div>{status_pill(state)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ---- Resolve MLflow experiment + latest run early so we can
        #      use epoch data in the header cards as well as charts.
        mlflow_exp = config.get("mlflow", {}).get("experiment_name", "")
        max_ep = config.get("training", {}).get("max_epochs", 0)
        current_epoch = 0
        mlflow_run_id = None
        mlflow_metrics = {}
        mlflow_error = None

        if mlflow_exp:
            try:
                runs = client.get_mlflow_runs(mlflow_exp, max_results=1)
                if runs:
                    latest = runs[0]
                    mlflow_metrics = latest.get("metrics", {})
                    mlflow_run_id = latest["run_id"]
                    epoch_history = client.get_run_metrics_history(mlflow_run_id, "epoch")
                    if epoch_history:
                        current_epoch = int(max(h["value"] for h in epoch_history))
                    elif "epoch" in mlflow_metrics:
                        current_epoch = int(mlflow_metrics["epoch"])
            except Exception as exc:
                mlflow_error = str(exc)

        c1, c2, c3 = st.columns(3)
        with c1:
            started = status.get("start_time")
            metric_card("Started", started.strftime("%H:%M:%S") if started else "—")
        with c2:
            dur = status.get("duration_seconds")
            if dur:
                metric_card("Duration", f"{dur / 60:.1f} min")
            elif started:
                elapsed = (datetime.now() - started).total_seconds() / 60
                metric_card("Elapsed", f"{elapsed:.1f} min")
            else:
                metric_card("Duration", "—")
        with c3:
            metric_card("Epochs", f"{current_epoch} / {max_ep}" if max_ep else "—")

        # Progress bar
        if state == "RUNNING" and max_ep:
            progress = min(current_epoch / max_ep, 1.0)
            st.progress(progress, text=f"Epoch {current_epoch} / {max_ep}")

        # MLflow experiment link
        if mlflow_exp:
            section_title("MLflow Experiment")
            try:
                host = client.workspace_client.config.host
                exp_id = client._resolve_experiment_id(mlflow_exp)
                if host and exp_id:
                    exp_url = f"{host.rstrip('/')}/#mlflow/experiments/{exp_id}"
                    st.markdown(
                        f'<div class="glass-card">'
                        f'<strong>Experiment:</strong> <code>{mlflow_exp}</code><br/>'
                        f'<a href="{exp_url}" target="_blank" style="color:#6C63FF;">Open in MLflow UI &rarr;</a>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                elif host:
                    st.markdown(
                        f'<div class="glass-card">'
                        f'<strong>Experiment:</strong> <code>{mlflow_exp}</code> '
                        f'<em>(not found on workspace — check the name/ID)</em></div>',
                        unsafe_allow_html=True,
                    )
            except Exception:
                st.markdown(f'<div class="glass-card"><strong>Experiment:</strong> <code>{mlflow_exp}</code></div>', unsafe_allow_html=True)

        # Run page link
        if status.get("run_page_url"):
            st.markdown(f"[Open Job Run in Databricks]({status['run_page_url']})")

        # Live loss / metric charts
        section_title("Training Curves")
        if mlflow_run_id:
            primary_metric = "eval_map" if task != "classification" else "eval_accuracy"
            loss_history = client.get_run_metrics_history(mlflow_run_id, "eval_loss")
            primary_history = client.get_run_metrics_history(mlflow_run_id, primary_metric)

            c1, c2 = st.columns(2)
            with c1:
                if loss_history:
                    fig = VisualizationHelper.training_metrics_chart(loss_history, "Eval Loss", title="Loss Curve")
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                elif "eval_loss" in mlflow_metrics:
                    metric_card("Eval Loss", f"{mlflow_metrics['eval_loss']:.4f}")
                else:
                    st.info("Waiting for loss data...")
            with c2:
                if primary_history:
                    fig = VisualizationHelper.training_metrics_chart(
                        primary_history, primary_metric, title=primary_metric.replace("_", " ").title()
                    )
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                elif primary_metric in mlflow_metrics:
                    metric_card(primary_metric.replace("_", " ").title(), f"{mlflow_metrics[primary_metric]:.4f}")
                else:
                    st.info("Waiting for metric data...")

            # Extra metrics snapshot
            if mlflow_metrics:
                section_title("Latest Metric Snapshot")
                display_keys = [k for k in sorted(mlflow_metrics.keys()) if k.startswith("eval_")][:8]
                if display_keys:
                    cols = st.columns(min(len(display_keys), 4))
                    for i, k in enumerate(display_keys):
                        with cols[i % 4]:
                            metric_card(k.replace("eval_", "").replace("_", " ").title(), f"{mlflow_metrics[k]:.4f}")
        elif mlflow_exp:
            if mlflow_error:
                st.warning(f"Could not fetch MLflow data: {mlflow_error}")
            else:
                st.info("No MLflow runs found yet — metrics will appear once training starts logging.")
        else:
            st.info("Configure an MLflow experiment in your YAML to see live charts.")

        st.markdown("")
        c1, c2, c3 = st.columns(3)
        with c1:
            if state == "RUNNING" and st.button("Cancel Training", use_container_width=True):
                if client.cancel_job(active_run_id):
                    st.success("Cancelled")
                    StateManager.clear_active_training_run()
                    st.rerun()
        with c2:
            if st.button("Refresh", type="primary", use_container_width=True):
                st.rerun()
        with c3:
            if state == "TERMINATED" and st.button("Clear Run", use_container_width=True):
                StateManager.clear_active_training_run()
                st.rerun()

        auto_refresh = st.checkbox("Auto-refresh every 30s", value=False)
        if auto_refresh and state == "RUNNING":
            time.sleep(30)
            st.rerun()


# =========================== TAB 3 — History ===============================
with tab_history:
    section_title("Past Runs")
    history = StateManager.get("training_history", [])

    if not history:
        st.info("No training history yet.")
    else:
        import pandas as pd

        rows = [
            {
                "Job": r.get("job_name", "—"),
                "Task": r.get("task", "—"),
                "Model": r.get("model", "—").split("/")[-1],
                "Status": r.get("status", "—"),
                "Date": r.get("timestamp", "—")[:19],
                "Run ID": r.get("run_id", "—")[:12],
            }
            for r in history
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        for idx, run in enumerate(history):
            with st.expander(f"{run.get('job_name', f'Run {idx}')}"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**Run ID:** `{run.get('run_id', '—')}`")
                    st.markdown(f"**Job ID:** `{run.get('job_id', '—')}`")
                with c2:
                    st.markdown(f"**Model:** {run.get('model', '—')}")
                    st.markdown(f"**Config:** `{Path(run.get('config_path', '—')).name}`")
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("View Metrics", key=f"h_met_{idx}", use_container_width=True):
                        st.switch_page("pages/4_📈_Evaluation.py")
                with c2:
                    if st.button("Set Active", key=f"h_act_{idx}", use_container_width=True):
                        StateManager.set_active_training_run(run.get("run_id"))
                        st.rerun()
                with c3:
                    if st.button("Remove", key=f"h_rm_{idx}", use_container_width=True):
                        history.remove(run)
                        StateManager.set("training_history", history)
                        st.rerun()


# =========================== Sidebar ========================================
with st.sidebar:
    st.markdown("### Training")
    active = StateManager.get_active_training_run()
    if active:
        st.markdown(f'{status_pill("RUNNING")}', unsafe_allow_html=True)
        st.code(f"Run: {str(active)[:16]}...")
    else:
        st.info("No active run")
    st.divider()
    st.metric("Total Runs", len(StateManager.get("training_history", [])))
