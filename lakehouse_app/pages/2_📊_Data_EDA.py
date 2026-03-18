"""
Page 2 — Data Exploration & Analysis
Visualize images with annotations, class distributions, and data quality.
"""

import streamlit as st
import random
from collections import Counter
from typing import Dict, List

from utils.state_manager import StateManager
from utils.databricks_client import DatabricksJobClient
from components.theme import inject_theme, page_header, metric_card, section_title
from components.image_viewer import ImageViewer
from components.visualizations import VisualizationHelper

inject_theme()
StateManager.initialize()

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


@st.cache_resource
def _client():
    return DatabricksJobClient()


client = _client()

page_header("Data Explorer", "Visualize your dataset, inspect annotations, and analyse distributions")

config = StateManager.get_current_config()
if not config:
    st.warning("No active configuration — head to **Config Setup** first.")
    if st.button("Open Config Setup"):
        st.switch_page("pages/1_⚙️_Config_Setup.py")
    st.stop()

task = config.get("model", {}).get("task_type", "detection")
data_cfg = config.get("data", {})
train_path = data_cfg.get("train_data_path", "")
val_path = data_cfg.get("val_data_path", "")
train_ann = data_cfg.get("train_annotation_file", "")
val_ann = data_cfg.get("val_annotation_file", "")
num_classes = config.get("model", {}).get("num_classes", 0)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_annotated, tab_dist, tab_quality = st.tabs(
    ["Overview", "Annotated Samples", "Class Distribution", "Data Quality"]
)

# ======================= TAB 1 — Overview ==================================
with tab_overview:
    section_title("Dataset Paths")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="glass-card"><strong>Training</strong><br/><code>{train_path}</code></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="glass-card"><strong>Validation</strong><br/><code>{val_path}</code></div>', unsafe_allow_html=True)

    if st.button("Scan Dataset", type="primary"):
        with st.spinner("Reading volumes..."):
            def _count(dir_path):
                return len(client.list_volume_files(dir_path, extensions=IMAGE_EXTS)) if dir_path else 0

            def _ann_count(ann_file):
                data = client.download_volume_json(ann_file)
                return len(data.get("annotations", [])) if data else 0

            train_imgs = _count(train_path)
            val_imgs = _count(val_path)
            train_anns = _ann_count(train_ann) if train_ann else 0
            val_anns = _ann_count(val_ann) if val_ann else 0
            total = train_imgs + val_imgs

        st.markdown("")
        cols = st.columns(4)
        with cols[0]:
            metric_card("Train Images", f"{train_imgs:,}")
        with cols[1]:
            metric_card("Val Images", f"{val_imgs:,}")
        with cols[2]:
            metric_card("Total Annotations", f"{train_anns + val_anns:,}")
        with cols[3]:
            pct = f"{val_imgs / total * 100:.0f}%" if total else "—"
            metric_card("Val Split", pct)

        st.markdown("")
        fig = VisualizationHelper.class_distribution_chart(
            {"Train": train_imgs, "Val": val_imgs},
            "Dataset Split",
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        img_size = data_cfg.get("image_size", "N/A")
        if isinstance(img_size, list):
            img_size = f"{img_size[0]} x {img_size[1]}"
        st.info(f"Images will be resized to **{img_size}** during training")


# ======================= TAB 2 — Annotated Samples =========================
with tab_annotated:
    section_title("Images with Annotations")

    split = st.selectbox("Split", ["Training", "Validation"], key="ann_split")
    n_samples = st.slider("Samples", 1, 12, 4, key="ann_n")

    if st.button("Load Annotated Samples", type="primary", key="load_annotated"):
        split_path = train_path if split == "Training" else val_path
        ann_file = train_ann if split == "Training" else val_ann

        if not split_path:
            st.warning("Path not configured for this split")
        else:
            with st.spinner("Downloading images & annotations..."):
                all_files = client.list_volume_files(split_path, extensions=IMAGE_EXTS)
                if not all_files:
                    st.warning("No image files found")
                else:
                    sample_files = random.sample(all_files, min(n_samples, len(all_files)))
                    coco_data = None
                    if task == "detection" and ann_file:
                        coco_data = client.download_volume_json(ann_file)

                    cat_map = {}
                    img_id_map = {}
                    ann_by_img = {}
                    if coco_data:
                        cat_map = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
                        img_id_map = {img["file_name"]: img["id"] for img in coco_data.get("images", [])}
                        for a in coco_data.get("annotations", []):
                            ann_by_img.setdefault(a["image_id"], []).append(a)

                    images, captions = [], []
                    for fname in sample_files:
                        full = split_path.rstrip("/") + "/" + fname
                        img = client.download_volume_image(full)
                        if img is None:
                            continue

                        if task == "detection" and coco_data:
                            img_id = img_id_map.get(fname)
                            anns = ann_by_img.get(img_id, [])
                            if anns:
                                boxes = [a["bbox"] for a in anns]
                                labels = [cat_map.get(a["category_id"], str(a["category_id"])) for a in anns]
                                img = ImageViewer.draw_bounding_boxes(img, boxes, labels=labels, format="xywh")
                            captions.append(f"{fname}  ({len(anns)} annotations)")
                        elif task == "classification":
                            captions.append(fname)
                        else:
                            captions.append(fname)

                        images.append(img)

                    if images:
                        cols_per_row = min(len(images), 4)
                        rows = [images[i:i + cols_per_row] for i in range(0, len(images), cols_per_row)]
                        cap_rows = [captions[i:i + cols_per_row] for i in range(0, len(captions), cols_per_row)]
                        for img_row, cap_row in zip(rows, cap_rows):
                            cols = st.columns(cols_per_row)
                            for idx, (im, cap) in enumerate(zip(img_row, cap_row)):
                                with cols[idx]:
                                    st.image(im, caption=cap, use_container_width=True)
                    else:
                        st.warning("Could not download any images")

    if task == "classification":
        st.info("Classification uses an ImageFolder layout — class is inferred from directory name.")


# ======================= TAB 3 — Class Distribution ========================
with tab_dist:
    section_title("Category Analysis")

    if st.button("Analyse Distribution", type="primary", key="analyse_dist"):
        class_counts: Dict[str, int] = {}
        with st.spinner("Reading annotations..."):
            if task == "detection" and train_ann:
                coco_data = client.download_volume_json(train_ann)
                if coco_data:
                    cat_map = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
                    counter = Counter(a["category_id"] for a in coco_data.get("annotations", []))
                    class_counts = {cat_map.get(cid, str(cid)): cnt for cid, cnt in counter.most_common()}
            elif task == "classification" and train_path:
                subdirs = client.list_volume_dirs(train_path)
                for d in sorted(subdirs):
                    n = len(client.list_volume_files(train_path.rstrip("/") + "/" + d, extensions=IMAGE_EXTS))
                    if n > 0:
                        class_counts[d] = n

        if not class_counts:
            st.info("Could not read real annotations. Using sample distribution.")
            random.seed(42)
            class_counts = {f"class_{i}": random.randint(80, 900) for i in range(min(num_classes or 10, 20))}

        total = sum(class_counts.values())
        mean_count = total / len(class_counts) if class_counts else 0
        sorted_items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        max_cls, max_cnt = sorted_items[0]
        min_cls, min_cnt = sorted_items[-1]
        ratio = max_cnt / min_cnt if min_cnt else float("inf")

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Classes", str(len(class_counts)))
        with c2:
            metric_card("Total Instances", f"{total:,}")
        with c3:
            metric_card("Most Common", f"{max_cls} ({max_cnt:,})")
        with c4:
            metric_card("Imbalance Ratio", f"{ratio:.1f}x")

        st.markdown("")

        # Bar chart
        import plotly.graph_objects as go

        fig = go.Figure()
        names = [item[0] for item in sorted_items]
        counts = [item[1] for item in sorted_items]
        colors = ["#FF6B6B" if c < mean_count * 0.5 else "#FFAA00" if c < mean_count * 0.8 else "#00D68F" for c in counts]
        fig.add_trace(go.Bar(x=names, y=counts, marker_color=colors, text=counts, textposition="outside"))
        fig.update_layout(
            title="Instance Count per Category",
            xaxis_title="Category", yaxis_title="Count",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-45,
            margin=dict(b=120),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Over / under represented analysis
        section_title("Representation Analysis")
        over = {k: v for k, v in class_counts.items() if v > mean_count * 1.5}
        under = {k: v for k, v in class_counts.items() if v < mean_count * 0.5}

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="glass-card">'
                f'<strong style="color:#00D68F;">Over-represented ({len(over)})</strong>'
                f'<p style="color:#8B949E;font-size:0.85rem;">Categories with &gt;1.5x the mean count</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if over:
                for name, cnt in sorted(over.items(), key=lambda x: -x[1]):
                    pct = cnt / total * 100
                    st.markdown(f"**{name}** — {cnt:,} instances ({pct:.1f}%)")
            else:
                st.success("No over-represented categories")
        with c2:
            st.markdown(
                f'<div class="glass-card">'
                f'<strong style="color:#FF6B6B;">Under-represented ({len(under)})</strong>'
                f'<p style="color:#8B949E;font-size:0.85rem;">Categories with &lt;0.5x the mean count</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if under:
                for name, cnt in sorted(under.items(), key=lambda x: x[1]):
                    pct = cnt / total * 100
                    st.markdown(f"**{name}** — {cnt:,} instances ({pct:.1f}%)")
            else:
                st.success("No under-represented categories")

        if ratio > 10:
            st.warning(f"Severe class imbalance ({ratio:.1f}x). Consider using class weights or over-sampling.")
        elif ratio > 5:
            st.info(f"Moderate imbalance ({ratio:.1f}x). Data augmentation recommended for tail classes.")
        else:
            st.success(f"Classes are relatively balanced (ratio {ratio:.1f}x).")


# ======================= TAB 4 — Data Quality ==============================
with tab_quality:
    section_title("Validation Checks")

    if st.button("Run Validation", type="primary", key="validate"):
        checks = []
        with st.spinner("Validating..."):
            # Path checks
            for label, path in [("Train images", train_path), ("Val images", val_path)]:
                if path:
                    files = client.list_volume_files(path, extensions=IMAGE_EXTS)
                    checks.append((label, len(files) > 0, f"{len(files)} files"))
                else:
                    checks.append((label, False, "Path not set"))

            if task == "detection":
                for label, ann_path in [("Train annotations", train_ann), ("Val annotations", val_ann)]:
                    if ann_path:
                        data = client.download_volume_json(ann_path)
                        ok = data is not None and "annotations" in data
                        count = len(data.get("annotations", [])) if data else 0
                        checks.append((label, ok, f"{count} annotations"))
                    else:
                        checks.append((label, False, "Not configured"))

        passed = sum(1 for _, ok, _ in checks if ok)
        total = len(checks)
        st.markdown("")
        c1, c2 = st.columns([1, 3])
        with c1:
            metric_card("Checks Passed", f"{passed}/{total}")
        with c2:
            for label, ok, detail in checks:
                icon = "✅" if ok else "❌"
                st.markdown(f"{icon} **{label}** — {detail}")

        st.markdown("")
        section_title("Recommendations")
        img_size = data_cfg.get("image_size", None)
        batch = data_cfg.get("batch_size", None)
        recs = []
        if img_size:
            recs.append(f"Image size set to **{img_size}** — appropriate for the selected model")
        if batch:
            recs.append(f"Batch size **{batch}** — adjust if you see OOM errors")
        if not data_cfg.get("augment"):
            recs.append("Data augmentation is **disabled** — enable it for better generalisation")
        for r in recs:
            st.info(r)


# ======================= Sidebar ============================================
with st.sidebar:
    st.markdown(f"### Dataset Info")
    st.markdown(f"**Task:** {task.replace('_',' ').title()}")
    st.markdown(f"**Image Size:** {data_cfg.get('image_size', 'N/A')}")
    st.markdown(f"**Batch Size:** {data_cfg.get('batch_size', 'N/A')}")
    st.divider()
    if st.button("Start Training", use_container_width=True):
        st.switch_page("pages/3_🚀_Training.py")
    if st.button("Edit Config", use_container_width=True):
        st.switch_page("pages/1_⚙️_Config_Setup.py")
