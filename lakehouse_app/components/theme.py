"""
Global theme and CSS for the CV Pipeline App.
Injects a futuristic dark design system via st.markdown.
"""

import streamlit as st

ACCENT = "#6C63FF"
ACCENT_LIGHT = "#A29BFE"
SUCCESS = "#00D68F"
WARNING = "#FFAA00"
DANGER = "#FF6B6B"
BG_CARD = "rgba(22, 27, 34, 0.8)"
BORDER = "rgba(108, 99, 255, 0.25)"
TEXT_DIM = "#8B949E"

GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="st-"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}

/* ---- sidebar ---- */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D1117 0%, #161B22 100%);
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {{
    color: {ACCENT_LIGHT};
}}

/* ---- glass card ---- */
.glass-card {{
    background: {BG_CARD};
    backdrop-filter: blur(12px);
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
.glass-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
}}

/* ---- metric card ---- */
.metric-card {{
    background: linear-gradient(135deg, rgba(108,99,255,0.15) 0%, rgba(162,155,254,0.08) 100%);
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}}
.metric-card .metric-value {{
    font-size: 2rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.2;
}}
.metric-card .metric-label {{
    font-size: 0.8rem;
    font-weight: 500;
    color: {TEXT_DIM};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.4rem;
}}
.metric-card .metric-delta {{
    font-size: 0.85rem;
    margin-top: 0.25rem;
}}
.metric-delta.positive {{ color: {SUCCESS}; }}
.metric-delta.negative {{ color: {DANGER}; }}

/* ---- status pill ---- */
.status-pill {{
    display: inline-block;
    padding: 0.25rem 0.85rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}}
.status-pill.ready   {{ background: rgba(0,214,143,0.15); color: {SUCCESS}; border: 1px solid rgba(0,214,143,0.3); }}
.status-pill.running {{ background: rgba(255,170,0,0.15); color: {WARNING}; border: 1px solid rgba(255,170,0,0.3); }}
.status-pill.failed  {{ background: rgba(255,107,107,0.15); color: {DANGER}; border: 1px solid rgba(255,107,107,0.3); }}
.status-pill.pending {{ background: rgba(139,148,158,0.15); color: {TEXT_DIM}; border: 1px solid rgba(139,148,158,0.3); }}

/* ---- page header ---- */
.page-header {{
    padding: 1.5rem 0 1rem 0;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 2rem;
}}
.page-header h1 {{
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_LIGHT} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}}
.page-header p {{
    color: {TEXT_DIM};
    font-size: 0.95rem;
    margin: 0.3rem 0 0 0;
}}

/* ---- section divider ---- */
.section-title {{
    font-size: 1.1rem;
    font-weight: 600;
    color: {ACCENT_LIGHT};
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(108,99,255,0.15);
}}

/* ---- tabs override ---- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.5rem;
    background: transparent;
}}
.stTabs [data-baseweb="tab"] {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.5rem 1.2rem;
    color: {TEXT_DIM};
    font-weight: 500;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(108,99,255,0.25) 0%, rgba(162,155,254,0.12) 100%) !important;
    border-color: {ACCENT} !important;
    color: #FFFFFF !important;
}}

/* ---- buttons ---- */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(135deg, {ACCENT} 0%, #5A52D5 100%);
    border: none;
    border-radius: 10px;
    color: white;
    font-weight: 600;
    transition: all 0.2s ease;
}}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {{
    box-shadow: 0 4px 20px rgba(108,99,255,0.35);
    transform: translateY(-1px);
}}

/* ---- dataframes ---- */
.stDataFrame {{
    border: 1px solid {BORDER};
    border-radius: 12px;
    overflow: hidden;
}}

/* ---- expander ---- */
details {{
    border: 1px solid {BORDER} !important;
    border-radius: 12px !important;
    background: {BG_CARD} !important;
}}

/* ---- hero banner ---- */
.hero-banner {{
    background: linear-gradient(135deg, rgba(108,99,255,0.2) 0%, rgba(162,155,254,0.05) 60%, rgba(0,214,143,0.08) 100%);
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
}}
.hero-banner h1 {{
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #FFFFFF 0%, {ACCENT_LIGHT} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem 0;
}}
.hero-banner p {{
    color: {TEXT_DIM};
    font-size: 1.05rem;
    max-width: 640px;
    margin: 0 auto;
}}

/* ---- nav card ---- */
.nav-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 1.5rem;
    cursor: pointer;
    transition: all 0.25s ease;
    height: 100%;
}}
.nav-card:hover {{
    border-color: {ACCENT};
    box-shadow: 0 8px 32px rgba(108,99,255,0.12);
    transform: translateY(-3px);
}}
.nav-card .nav-icon {{
    font-size: 1.6rem;
    margin-bottom: 0.6rem;
}}
.nav-card h3 {{
    font-size: 1rem;
    font-weight: 600;
    color: #E6EDF3;
    margin: 0 0 0.4rem 0;
}}
.nav-card p {{
    font-size: 0.82rem;
    color: {TEXT_DIM};
    margin: 0;
    line-height: 1.5;
}}

/* ---- endpoint card ---- */
.endpoint-card {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}}
.endpoint-card .endpoint-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}}
.endpoint-card .endpoint-name {{
    font-size: 1.1rem;
    font-weight: 600;
    color: #E6EDF3;
}}
.endpoint-card .detail-row {{
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(139,148,158,0.1);
    font-size: 0.88rem;
}}
.endpoint-card .detail-label {{ color: {TEXT_DIM}; }}
.endpoint-card .detail-value {{ color: #E6EDF3; font-weight: 500; }}

/* ---- image grid ---- */
.img-annotated {{
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid {BORDER};
}}

/* ---- progress ring ---- */
.progress-container {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 0;
}}
</style>
"""


def inject_theme():
    """Call once per page to inject the global theme."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    """Render a styled page header."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="page-header"><h1>{title}</h1>{sub}</div>', unsafe_allow_html=True)


def metric_card(label: str, value, delta: str = "", delta_positive: bool = True):
    """Render a single metric card."""
    delta_html = ""
    if delta:
        cls = "positive" if delta_positive else "negative"
        delta_html = f'<div class="metric-delta {cls}">{delta}</div>'
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'{delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def status_pill(status: str):
    """Return HTML for a status pill."""
    status_upper = status.upper()
    mapping = {
        "READY": "ready", "NOT_UPDATING": "ready", "SUCCESS": "ready", "FINISHED": "ready",
        "RUNNING": "running", "PENDING": "running", "UPDATING": "running",
        "FAILED": "failed", "ERROR": "failed", "CANCELLED": "failed", "TERMINATED": "failed",
    }
    cls = mapping.get(status_upper, "pending")
    return f'<span class="status-pill {cls}">{status}</span>'


def section_title(text: str):
    """Render a styled section title."""
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)
