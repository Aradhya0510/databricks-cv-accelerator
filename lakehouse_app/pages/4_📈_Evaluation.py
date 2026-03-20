"""
Page 4 — Evaluation
Metrics dashboard, prediction overlays on images, and model comparison.
"""

import streamlit as st
from datetime import datetime

from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient
from components.theme import inject_theme, page_header, metric_card, section_title, status_pill
from components.visualizations import VisualizationHelper
from components.metrics_display import MetricsDisplay
from components.image_viewer import ImageViewer

inject_theme()
StateManager.initialize()

page_header("Evaluation", "Inspect metrics, view predictions on images, and compare runs")

client = DatabricksJobClient()
config = StateManager.get_current_config()

# ---------------------------------------------------------------------------
tab_dash, tab_preds, tab_compare, tab_reports = st.tabs(
    ["Metrics Dashboard", "Prediction Viewer", "Model Comparison", "Reports"]
)

# =========================== TAB 1 — Metrics Dashboard =====================
with tab_dash:
    default_exp = (config or {}).get("mlflow", {}).get("experiment_name", "")

    c1, c2 = st.columns([4, 1])
    with c1:
        experiment_name = st.text_input("MLflow Experiment", value=default_exp, key="eval_exp")
    with c2:
        st.markdown("<br/>", unsafe_allow_html=True)
        load = st.button("Load Runs", type="primary", use_container_width=True, key="load_eval_runs")

    if load or st.session_state.get("_eval_runs"):
        with st.spinner("Fetching runs..."):
            runs = client.get_mlflow_runs(experiment_name, max_results=50) if load else st.session_state.get("_eval_runs", [])
            if load:
                st.session_state["_eval_runs"] = runs

        if not runs:
            st.info("No runs found.")
        else:
            run_map = {f"{r['run_name']} ({r['run_id'][:8]}) — {r['status']}": r for r in runs}
            sel = st.selectbox("Select Run", list(run_map.keys()), key="eval_run_sel")
            run = run_map[sel]
            metrics = run.get("metrics", {})
            task = run.get("params", {}).get("task", config.get("model", {}).get("task_type", "detection") if config else "detection")

            # Key metrics banner
            if task == "detection":
                keys = ["eval_map", "eval_map_50", "eval_map_75", "eval_loss"]
            elif task == "classification":
                keys = ["eval_accuracy", "eval_f1", "eval_precision", "eval_loss"]
            else:
                keys = ["eval_loss"]

            available = [k for k in keys if k in metrics]
            if available:
                cols = st.columns(len(available))
                for i, k in enumerate(available):
                    with cols[i]:
                        metric_card(k.replace("eval_", "").replace("_", " ").title(), f"{metrics[k]:.4f}")

            st.markdown("")

            # Metric history charts
            section_title("Metric Curves")
            all_metric_keys = sorted(metrics.keys())
            chosen = st.multiselect("Metrics to plot", all_metric_keys, default=available[:4], key="eval_chart_sel")

            if chosen:
                chart_cols = st.columns(min(len(chosen), 2))
                for idx, mk in enumerate(chosen):
                    with chart_cols[idx % 2]:
                        history = client.get_run_metrics_history(run["run_id"], mk)
                        if history:
                            fig = VisualizationHelper.training_metrics_chart(history, mk, title=mk.replace("_", " ").title())
                            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No history for {mk}")

            # Per-class metrics
            per_class = {k: v for k, v in metrics.items() if "class_" in k}
            if per_class:
                section_title("Per-Class Performance")
                import plotly.graph_objects as go

                names = [k.split("_")[-1] for k in sorted(per_class.keys())]
                vals = [per_class[k] for k in sorted(per_class.keys())]
                fig = go.Figure(go.Bar(x=names, y=vals, marker_color="#6C63FF"))
                fig.update_layout(
                    title="Per-Class Metric",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Params
            with st.expander("Run Parameters"):
                import pandas as pd

                params = run.get("params", {})
                if params:
                    st.dataframe(
                        pd.DataFrame([{"Parameter": k, "Value": v} for k, v in sorted(params.items())]),
                        use_container_width=True, hide_index=True,
                    )

    else:
        st.info("Enter an experiment name and click **Load Runs**.")


# =========================== TAB 2 — Prediction Viewer =====================
with tab_preds:
    section_title("Predicted Annotations on Images")

    st.info("Load evaluation results from `jobs/evaluate.py` to see predictions overlaid on images.")

    results_dir = (config or {}).get("output", {}).get("results_dir", "/tmp/results")
    rd = st.text_input("Results Directory", value=results_dir, key="pred_rd")

    if st.button("Load Evaluation Results", type="primary", key="load_eval_preds"):
        rd = rd.rstrip("/")
        eval_metrics = client.read_json(f"{rd}/evaluation_metrics.json")
        error_data = client.read_json(f"{rd}/error_analysis.json")
        bench_data = client.read_json(f"{rd}/benchmark.json")

        if eval_metrics:
            section_title("Evaluation Metrics")
            key_display = ["eval_map", "eval_map_50", "eval_map_75", "eval_loss",
                           "eval_accuracy", "eval_f1", "eval_precision", "eval_recall"]
            found = [k for k in key_display if k in eval_metrics]
            if found:
                cols = st.columns(min(len(found), 4))
                for i, k in enumerate(found):
                    with cols[i % 4]:
                        metric_card(k.replace("eval_", "").replace("_", " ").title(), f"{eval_metrics[k]:.4f}")

            per_class = {k: v for k, v in eval_metrics.items() if "map_class_" in k}
            if per_class:
                import plotly.graph_objects as go
                names = [k.split("_")[-1] for k in sorted(per_class.keys())]
                vals = [per_class[k] for k in sorted(per_class.keys())]
                fig = go.Figure(go.Bar(x=names, y=vals, marker_color="#6C63FF"))
                fig.update_layout(
                    title="Per-Class Average Precision",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No `evaluation_metrics.json` found in `{rd}`")

        if error_data:
            section_title("Error Analysis")
            summary = error_data.get("summary", {})
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("True Positives", str(summary.get("true_positives", 0)))
            with c2:
                fp = (summary.get("false_positives_background", 0)
                      + summary.get("false_positives_confusion", 0)
                      + summary.get("false_positives_localisation", 0))
                metric_card("False Positives", str(fp))
            with c3:
                metric_card("False Negatives", str(summary.get("false_negatives", 0)))
            with c4:
                total = summary.get("true_positives", 0) + fp + summary.get("false_negatives", 0)
                prec = summary.get("true_positives", 0) / (summary.get("true_positives", 0) + fp) if (summary.get("true_positives", 0) + fp) > 0 else 0
                metric_card("Precision", f"{prec:.2%}")

            # FP breakdown
            import plotly.graph_objects as go

            fp_bg = summary.get("false_positives_background", 0)
            fp_conf = summary.get("false_positives_confusion", 0)
            fp_loc = summary.get("false_positives_localisation", 0)
            if fp_bg or fp_conf or fp_loc:
                fig = go.Figure(go.Pie(
                    labels=["Background", "Confusion", "Localisation"],
                    values=[fp_bg, fp_conf, fp_loc],
                    marker_colors=["#FF6B6B", "#FFAA00", "#6C63FF"],
                    hole=0.4,
                ))
                fig.update_layout(
                    title="False Positive Breakdown",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True)

        if bench_data:
            section_title("Benchmark")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("FPS", f"{bench_data.get('fps', 0):.1f}")
            with c2:
                lat = bench_data.get("latency_per_batch_ms", {})
                metric_card("P50 Latency", f"{lat.get('p50', 0):.0f} ms")
            with c3:
                metric_card("P95 Latency", f"{lat.get('p95', 0):.0f} ms")
            with c4:
                mem = bench_data.get("gpu_memory_mb", 0)
                metric_card("GPU Memory", f"{mem:.0f} MB" if mem else "—")


# =========================== TAB 3 — Model Comparison ======================
with tab_compare:
    section_title("Compare Runs Side-by-Side")

    exp = st.text_input("Experiment", value=(config or {}).get("mlflow", {}).get("experiment_name", ""), key="cmp_exp")

    if st.button("Load Runs", type="primary", key="cmp_load"):
        with st.spinner("Loading..."):
            runs = client.get_mlflow_runs(exp, max_results=50)

        if not runs:
            st.info("No runs found.")
        else:
            run_map = {f"{r['run_name']} ({r['run_id'][:8]})": r for r in runs}
            sel = st.multiselect("Select runs", list(run_map.keys()), default=list(run_map.keys())[:3], key="cmp_sel")
            if sel:
                selected = [run_map[s] for s in sel]
                all_m = set()
                for r in selected:
                    all_m.update(r.get("metrics", {}).keys())
                met_sel = st.multiselect("Metrics", sorted(all_m), default=sorted(all_m)[:4], key="cmp_met")

                if met_sel:
                    MetricsDisplay.display_comparison_table(selected, met_sel)
                    st.markdown("")
                    for m in met_sel:
                        fig = VisualizationHelper.model_comparison_chart(selected, m)
                        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                        st.plotly_chart(fig, use_container_width=True)


# =========================== TAB 4 — Reports ===============================
with tab_reports:
    section_title("Generate Report")

    c1, c2 = st.columns(2)
    with c1:
        report_type = st.selectbox("Report Type", ["Full Report", "Performance Summary", "Error Analysis"], key="rpt_type")
    with c2:
        report_format = st.selectbox("Format", ["JSON", "Markdown"], key="rpt_fmt")

    if st.button("Generate", type="primary", key="gen_report"):
        import json as _json

        cfg = StateManager.get_current_config() or {}
        exp_name = cfg.get("mlflow", {}).get("experiment_name", "")
        report = {"generated_at": datetime.now().isoformat(), "type": report_type}
        if exp_name:
            try:
                runs = client.get_mlflow_runs(exp_name, max_results=10)
                report["runs"] = [
                    {"run_id": r["run_id"], "name": r["run_name"], "status": r["status"], "metrics": r.get("metrics", {})}
                    for r in runs
                ]
            except Exception:
                report["runs"] = []

        rd = cfg.get("output", {}).get("results_dir", "")
        for fname in ["evaluation_metrics.json", "error_analysis.json", "benchmark.json"]:
            if rd:
                data = client.read_json(f"{rd.rstrip('/')}/{fname}")
                if data:
                    report[fname.replace(".json", "")] = data

        if report_format == "JSON":
            content = _json.dumps(report, indent=2, default=str)
            mime = "application/json"
            ext = "json"
        else:
            lines = [f"# Evaluation Report — {report_type}", f"Generated: {report['generated_at']}", ""]
            for r in report.get("runs", []):
                lines.append(f"## {r['name']} ({r['status']})")
                for k, v in sorted(r.get("metrics", {}).items()):
                    lines.append(f"- **{k}:** {v:.4f}" if isinstance(v, float) else f"- **{k}:** {v}")
                lines.append("")
            content = "\n".join(lines)
            mime = "text/markdown"
            ext = "md"

        st.success("Report generated")
        st.download_button("Download Report", data=content, file_name=f"report_{datetime.now():%Y%m%d_%H%M%S}.{ext}", mime=mime)


# =========================== Sidebar ========================================
with st.sidebar:
    st.markdown("### Evaluation")
    if config:
        t = config.get("model", {}).get("task_type", "—")
        st.markdown(f"**Task:** {t.replace('_', ' ').title()}")
        key_m = {"detection": "mAP, mAP@50, mAP@75", "classification": "Accuracy, F1, Precision"}
        st.markdown(f"**Metrics:** {key_m.get(t, 'Loss')}")
    st.divider()
    if st.button("Train New Model", use_container_width=True):
        st.switch_page("pages/3_🚀_Training.py")
