"""
Global theme — ML Accelerator Design System v1.0
Dark-first, data-native design language for the CV Pipeline App.
"""

import streamlit as st

# --------------------------------------------------------------------------- #
# Colour tokens
# --------------------------------------------------------------------------- #

# Background tiers (four levels of elevation)
BG_BASE = "#0D0F12"
BG_SURFACE = "#141720"
BG_RAISED = "#1C2030"
BG_OVERLAY = "#232840"

# Semantic accents
ACCENT = "#00C2A8"
ACCENT_WARM = "#F4A742"
ACCENT_ALERT = "#F25C5C"
ACCENT_INFO = "#5B8AF5"

# Text levels
TEXT_PRIMARY = "#EDF0F7"
TEXT_SECONDARY = "#8A91A8"
TEXT_TERTIARY = "#4E566A"

# Borders
BORDER_SUBTLE = "rgba(255,255,255,0.06)"
BORDER_ACCENT = f"rgba(0,194,168,0.3)"

# --------------------------------------------------------------------------- #
# Global CSS
# --------------------------------------------------------------------------- #

GLOBAL_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@400;500;600;700&family=Figtree:wght@300;400;500;600&display=swap');

  html, body, [class*="st-"] {{
      font-family: 'Figtree', -apple-system, BlinkMacSystemFont, sans-serif;
      color: {TEXT_SECONDARY};
  }}
  h1, h2, h3, h4 {{
      font-family: 'Syne', sans-serif !important;
      color: {TEXT_PRIMARY} !important;
  }}
  code, pre {{
      font-family: 'IBM Plex Mono', monospace !important;
  }}

  /* ---- sidebar (200px, bg-base) ---- */
  section[data-testid="stSidebar"] {{
      width: 200px !important;
      min-width: 200px !important;
      max-width: 250px !important;
      background: {BG_BASE};
      border-right: 1px solid {BORDER_SUBTLE};
  }}
  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li[class*="active"] {{
      background: rgba(0,194,168,0.08);
      border-left: 2px solid {ACCENT};
  }}
  section[data-testid="stSidebar"] [data-testid="stSidebarNav"] li[class*="active"] span {{
      color: {ACCENT} !important;
  }}
  section[data-testid="stSidebar"] .stMarkdown {{
      font-family: 'Figtree', sans-serif;
      font-size: 13px;
  }}
  section[data-testid="stSidebar"] .stMarkdown h1,
  section[data-testid="stSidebar"] .stMarkdown h2,
  section[data-testid="stSidebar"] .stMarkdown h3 {{
      font-family: 'Syne', sans-serif !important;
      color: {TEXT_PRIMARY} !important;
  }}

  /* ---- st.metric override ---- */
  .stMetric label {{
      font-family: 'IBM Plex Mono', monospace !important;
      font-size: 10px !important;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: {TEXT_TERTIARY} !important;
  }}
  .stMetric [data-testid="stMetricValue"] {{
      font-family: 'Syne', sans-serif !important;
      font-weight: 700 !important;
      font-size: 28px !important;
      color: {TEXT_PRIMARY} !important;
  }}
  .stMetric [data-testid="stMetricDelta"] svg {{
      display: none;
  }}

  /* ---- metric card (custom) ---- */
  .metric-card {{
      background: {BG_RAISED};
      border: 1px solid {BORDER_SUBTLE};
      border-radius: 8px;
      padding: 16px 20px;
      text-align: center;
  }}
  .metric-card .metric-label {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 10px;
      font-weight: 400;
      color: {TEXT_TERTIARY};
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
  }}
  .metric-card .metric-value {{
      font-family: 'Syne', sans-serif;
      font-size: 28px;
      font-weight: 700;
      color: {TEXT_PRIMARY};
      line-height: 1.1;
  }}
  .metric-card .metric-delta {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 11px;
      margin-top: 4px;
  }}
  .metric-delta.positive {{ color: {ACCENT}; }}
  .metric-delta.negative {{ color: {ACCENT_ALERT}; }}

  /* ---- status badge ---- */
  .status-badge {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 4px 10px;
      border-radius: 4px;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 10px;
      font-weight: 400;
      text-transform: uppercase;
      letter-spacing: 0.05em;
  }}
  .status-badge .dot {{
      width: 5px;
      height: 5px;
      border-radius: 50%;
      display: inline-block;
  }}
  .status-badge.running  {{ background: rgba(0,194,168,0.10); color: {ACCENT}; }}
  .status-badge.running .dot  {{ background: {ACCENT}; }}
  .status-badge.queued   {{ background: rgba(244,167,66,0.10); color: {ACCENT_WARM}; }}
  .status-badge.queued .dot   {{ background: {ACCENT_WARM}; }}
  .status-badge.failed   {{ background: rgba(242,92,92,0.10); color: {ACCENT_ALERT}; }}
  .status-badge.failed .dot   {{ background: {ACCENT_ALERT}; }}
  .status-badge.registered {{ background: rgba(91,138,245,0.10); color: {ACCENT_INFO}; }}
  .status-badge.registered .dot {{ background: {ACCENT_INFO}; }}

  /* ---- progress bar ---- */
  .stProgress > div > div > div > div {{
      background: {ACCENT} !important;
      border-radius: 2px;
  }}
  .stProgress > div > div > div {{
      background: {BG_OVERLAY};
      height: 4px;
      border-radius: 2px;
  }}

  /* ---- code / config block ---- */
  .config-block {{
      background: {BG_RAISED};
      border-left: 2px solid {ACCENT};
      border-radius: 4px;
      padding: 16px;
      font-family: 'IBM Plex Mono', monospace;
      font-size: 12px;
      line-height: 1.7;
  }}

  /* ---- surface card ---- */
  .surface-card {{
      background: {BG_SURFACE};
      border: 1px solid {BORDER_SUBTLE};
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 16px;
  }}

  /* ---- raised card (inner panels) ---- */
  .raised-card {{
      background: {BG_RAISED};
      border: 1px solid {BORDER_SUBTLE};
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 8px;
  }}

  /* ---- page header (topbar) ---- */
  .page-header {{
      padding: 32px 0 16px 0;
      border-bottom: 1px solid {BORDER_SUBTLE};
      margin-bottom: 32px;
      display: flex;
      justify-content: space-between;
      align-items: center;
  }}
  .page-header .ph-left h1 {{
      font-family: 'Syne', sans-serif !important;
      font-size: 28px !important;
      font-weight: 700 !important;
      color: {TEXT_PRIMARY} !important;
      margin: 0 !important;
      line-height: 1.1;
  }}
  .page-header .ph-left p {{
      font-family: 'Figtree', sans-serif;
      color: {TEXT_SECONDARY};
      font-size: 14px;
      margin: 6px 0 0 0;
  }}

  /* ---- section heading ---- */
  .section-title {{
      font-family: 'Syne', sans-serif;
      font-size: 18px;
      font-weight: 600;
      color: {TEXT_PRIMARY};
      margin: 32px 0 16px 0;
      padding-bottom: 8px;
      border-bottom: 1px solid {BORDER_SUBTLE};
  }}

  /* ---- tabs ---- */
  .stTabs [data-baseweb="tab-list"] {{
      gap: 8px;
      background: transparent;
      border-bottom: 1px solid {BORDER_SUBTLE};
  }}
  .stTabs [data-baseweb="tab"] {{
      background: transparent;
      border: none;
      border-bottom: 2px solid transparent;
      border-radius: 0;
      padding: 8px 16px;
      color: {TEXT_SECONDARY};
      font-family: 'Figtree', sans-serif;
      font-weight: 500;
      font-size: 13px;
  }}
  .stTabs [aria-selected="true"] {{
      background: transparent !important;
      border-bottom: 2px solid {ACCENT} !important;
      color: {TEXT_PRIMARY} !important;
  }}

  /* ---- buttons ---- */
  .stButton > button {{
      font-family: 'Figtree', sans-serif !important;
      font-weight: 500;
      font-size: 13px;
      border-radius: 8px;
      padding: 8px 16px;
      transition: all 0.15s ease;
  }}
  .stButton > button[kind="primary"],
  .stButton > button[data-testid="stBaseButton-primary"] {{
      background: {ACCENT} !important;
      border: none !important;
      color: {BG_BASE} !important;
      font-weight: 600;
  }}
  .stButton > button[kind="primary"]:hover,
  .stButton > button[data-testid="stBaseButton-primary"]:hover {{
      opacity: 0.9;
      box-shadow: 0 2px 12px rgba(0,194,168,0.25);
  }}
  .stButton > button[kind="secondary"],
  .stButton > button[data-testid="stBaseButton-secondary"] {{
      background: {BG_RAISED} !important;
      border: 1px solid {BORDER_SUBTLE} !important;
      color: {TEXT_SECONDARY} !important;
  }}

  /* ---- data tables ---- */
  .stDataFrame {{
      border: 1px solid {BORDER_SUBTLE} !important;
      border-radius: 8px;
      overflow: hidden;
  }}
  .stDataFrame th {{
      font-family: 'IBM Plex Mono', monospace !important;
      font-size: 10px !important;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: {TEXT_TERTIARY} !important;
  }}
  .stDataFrame td {{
      font-family: 'Figtree', sans-serif;
      font-size: 12px;
      color: {TEXT_SECONDARY};
      border-bottom: 1px solid {BORDER_SUBTLE};
  }}

  /* ---- expander ---- */
  details {{
      border: 1px solid {BORDER_SUBTLE} !important;
      border-radius: 8px !important;
      background: {BG_SURFACE} !important;
  }}
  details summary {{
      font-family: 'Figtree', sans-serif;
      font-size: 14px;
      color: {TEXT_SECONDARY};
  }}

  /* ---- text input ---- */
  .stTextInput > div > div > input,
  .stSelectbox > div > div,
  .stNumberInput > div > div > input,
  .stTextArea > div > div > textarea {{
      background: {BG_RAISED} !important;
      border: 1px solid {BORDER_SUBTLE} !important;
      color: {TEXT_PRIMARY} !important;
      font-family: 'Figtree', sans-serif;
      border-radius: 8px;
  }}
  .stTextInput > div > div > input:focus,
  .stTextArea > div > div > textarea:focus {{
      border-color: {ACCENT} !important;
      box-shadow: 0 0 0 2px rgba(0,194,168,0.15) !important;
  }}

  /* ---- nav card ---- */
  .nav-card {{
      background: {BG_SURFACE};
      border: 1px solid {BORDER_SUBTLE};
      border-radius: 8px;
      padding: 24px;
      cursor: pointer;
      transition: all 0.2s ease;
      height: 100%;
  }}
  .nav-card:hover {{
      border-color: {ACCENT};
      box-shadow: 0 4px 16px rgba(0,194,168,0.08);
  }}
  .nav-card .nav-icon {{
      font-size: 1.4rem;
      margin-bottom: 8px;
  }}
  .nav-card h3 {{
      font-family: 'Syne', sans-serif !important;
      font-size: 14px !important;
      font-weight: 500 !important;
      color: {TEXT_PRIMARY} !important;
      margin: 0 0 8px 0 !important;
  }}
  .nav-card p {{
      font-family: 'Figtree', sans-serif;
      font-size: 12px;
      color: {TEXT_SECONDARY};
      margin: 0;
      line-height: 1.6;
  }}

  /* ---- endpoint card ---- */
  .endpoint-card {{
      background: {BG_SURFACE};
      border: 1px solid {BORDER_SUBTLE};
      border-radius: 8px;
      padding: 24px;
      margin-bottom: 16px;
  }}
  .endpoint-card .endpoint-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
  }}
  .endpoint-card .endpoint-name {{
      font-family: 'Syne', sans-serif;
      font-size: 16px;
      font-weight: 600;
      color: {TEXT_PRIMARY};
  }}
  .endpoint-card .detail-row {{
      display: flex;
      justify-content: space-between;
      padding: 8px 0;
      border-bottom: 1px solid {BORDER_SUBTLE};
      font-size: 12px;
  }}
  .endpoint-card .detail-label {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 11px;
      color: {TEXT_TERTIARY};
      text-transform: uppercase;
      letter-spacing: 0.05em;
  }}
  .endpoint-card .detail-value {{
      font-family: 'Figtree', sans-serif;
      color: {TEXT_PRIMARY};
      font-weight: 500;
  }}

  /* ---- hero banner ---- */
  .hero-banner {{
      background: {BG_SURFACE};
      border: 1px solid {BORDER_SUBTLE};
      border-radius: 8px;
      padding: 48px 32px;
      text-align: center;
      margin-bottom: 32px;
  }}
  .hero-banner h1 {{
      font-family: 'Syne', sans-serif !important;
      font-size: 28px !important;
      font-weight: 700 !important;
      color: {TEXT_PRIMARY} !important;
      margin: 0 0 8px 0 !important;
      line-height: 1.1;
  }}
  .hero-banner p {{
      font-family: 'Figtree', sans-serif;
      color: {TEXT_SECONDARY};
      font-size: 14px;
      max-width: 640px;
      margin: 0 auto;
      line-height: 1.6;
  }}

  /* ---- image annotation ---- */
  .img-annotated {{
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid {BORDER_SUBTLE};
  }}

  /* ---- live training pulse ---- */
  @keyframes pulse {{
      0%,100% {{ opacity: 1; }}
      50% {{ opacity: 0.3; }}
  }}

  /* ---- info / warning / error overrides ---- */
  .stAlert {{
      border-radius: 8px !important;
      font-family: 'Figtree', sans-serif;
      font-size: 13px;
  }}

  /* ---- review summary card ---- */
  .detail-row {{
      display: flex;
      justify-content: space-between;
      padding: 8px 0;
      border-bottom: 1px solid {BORDER_SUBTLE};
      font-size: 12px;
  }}
  .detail-label {{
      font-family: 'IBM Plex Mono', monospace;
      font-size: 11px;
      color: {TEXT_TERTIARY};
      text-transform: uppercase;
      letter-spacing: 0.05em;
  }}
  .detail-value {{
      font-family: 'Figtree', sans-serif;
      color: {TEXT_PRIMARY};
      font-weight: 500;
  }}
</style>
"""


def inject_theme():
    """Call once per page to inject the global theme."""
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", status: str = ""):
    """Render a styled page header with optional status badge on the right."""
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    badge = status_badge(status) if status else ""
    st.markdown(
        f'<div class="page-header">'
        f'<div class="ph-left"><h1>{title}</h1>{sub}</div>'
        f'<div class="ph-right">{badge}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value, delta: str = "", delta_positive: bool = True):
    """Render a single metric card following the design system."""
    delta_html = ""
    if delta:
        cls = "positive" if delta_positive else "negative"
        delta_html = f'<div class="metric-delta {cls}">{delta}</div>'
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'{delta_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


def status_badge(status: str) -> str:
    """Return HTML for a status badge with dot indicator + text label."""
    status_upper = status.upper()
    mapping = {
        "READY": "running", "NOT_UPDATING": "running", "SUCCESS": "running",
        "FINISHED": "running",
        "RUNNING": "queued", "PENDING": "queued", "UPDATING": "queued",
        "QUEUED": "queued",
        "FAILED": "failed", "ERROR": "failed", "CANCELLED": "failed",
        "TERMINATED": "failed",
        "REGISTERED": "registered",
    }
    cls = mapping.get(status_upper, "queued")
    return (
        f'<span class="status-badge {cls}">'
        f'<span class="dot"></span>{status}'
        f'</span>'
    )


# Keep backward compat for pages that call status_pill
status_pill = status_badge


def section_title(text: str):
    """Render a styled section heading (Syne 600, 18px)."""
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)
