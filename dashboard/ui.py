"""Shared look & feel for every FinIntel AI page.

Usage (right after st.set_page_config):
    from dashboard.ui import apply_theme, page_header, style_fig
    apply_theme()
    page_header("Title", "One-line description")
"""
import streamlit as st

# ── Design tokens ────────────────────────────────────────────
TEXT    = "inherit"
MUTED   = "#8b95a5"
BORDER  = "rgba(128,128,128,.28)"
SURFACE = "rgba(128,128,128,.07)"
ACCENT  = "#10a37f"
GOOD    = "#16a34a"
WARN    = "#d97706"
BAD     = "#dc2626"

SENTIMENT_COLORS = {"positive": GOOD, "negative": BAD, "neutral": WARN}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], [data-testid="stAppViewContainer"], .stMarkdown, button, input, textarea {{
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif !important;
}}

/* layout */
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }}
#MainMenu, footer, .stDeployButton, [data-testid="stDeployButton"] {{ display: none !important; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* type */
h1, h2, h3 {{ letter-spacing: -0.02em; color: {TEXT}; }}
h2 {{ font-weight: 650; }}
h3 {{ font-weight: 600; font-size: 1.15rem; margin-top: 1.6rem; }}

/* sidebar */
[data-testid="stSidebar"] {{
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebarNav"] a {{
    border-radius: 8px; padding: .35rem .6rem; margin: 1px 0;
}}
[data-testid="stSidebarNav"] a:hover {{ background: rgba(128,128,128,.15); }}
[data-testid="stSidebarNav"] a[aria-current="page"] {{
    background: rgba(128,128,128,.24); font-weight: 600;
}}

/* buttons */
.stButton > button {{
    border-radius: 10px; border: 1px solid {BORDER};
    font-weight: 500; transition: all .15s ease; box-shadow: none;
}}
.stButton > button:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
.stButton > button[kind="primary"] {{
    background: {ACCENT}; border-color: {ACCENT}; color: #fff;
}}
.stButton > button[kind="primary"]:hover {{ background: #0d8c6d; color: #fff; }}

/* inputs */
[data-baseweb="input"], [data-baseweb="select"] > div, textarea {{ border-radius: 10px !important; }}

/* metrics */
[data-testid="stMetric"] {{
    background: rgba(128,128,128,.06); border: 1px solid {BORDER}; border-radius: 12px; padding: 1rem 1.1rem;
}}
[data-testid="stMetricLabel"] {{ color: {MUTED}; }}

/* alerts, expanders, tabs, chat */
[data-testid="stAlert"] {{ border-radius: 10px; border: 1px solid {BORDER}; }}
[data-testid="stExpander"] {{ border: 1px solid {BORDER}; border-radius: 12px; background: rgba(128,128,128,.06); }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 8px 8px 0 0; padding: .5rem 1rem; }}
[data-testid="stChatMessage"] {{
    border: 1px solid {BORDER}; border-radius: 14px; background: rgba(128,128,128,.06); padding: 1rem 1.2rem;
}}
[data-testid="stFileUploaderDropzone"] {{
    border: 1.5px dashed rgba(128,128,128,.5); border-radius: 14px; background: {SURFACE};
}}
[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden; }}

/* custom components */
.fi-header {{ margin-bottom: 1.4rem; }}
.fi-eyebrow {{ color: {ACCENT}; font-size: .78rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }}
.fi-title {{ font-size: 2rem; font-weight: 700; letter-spacing: -0.03em; color: {TEXT}; margin: .15rem 0 .3rem; }}
.fi-sub {{ color: {MUTED}; font-size: 1.02rem; margin: 0; }}

.fi-card {{
    background: rgba(128,128,128,.06); border: 1px solid {BORDER}; border-radius: 14px;
    padding: 1.15rem 1.25rem; height: 100%; min-height: 170px; box-sizing: border-box;
}}
.fi-card.center {{ text-align: center; }}
.fi-card-title {{ font-weight: 600; color: {TEXT}; margin: 0 0 .25rem; }}
.fi-card-text  {{ color: {MUTED}; font-size: .88rem; margin: 0; line-height: 1.45; }}
.fi-step {{
    display:inline-flex; align-items:center; justify-content:center;
    width: 26px; height: 26px; border-radius: 8px; background: rgba(16,163,127,.18); color: {ACCENT};
    font-weight: 700; font-size: .8rem; margin-bottom: .6rem;
}}
.fi-pill {{
    display:inline-block; padding: .15rem .6rem; border-radius: 999px;
    font-size: .78rem; font-weight: 600;
}}
.fi-result {{
    background: rgba(128,128,128,.06); border:1px solid {BORDER}; border-left-width: 5px;
    border-radius: 14px; padding: 1.3rem 1.5rem; margin-bottom: 1rem;
}}
.fi-result h2 {{ margin: 0; }}
.fi-muted {{ color: {MUTED}; }}
.fi-article {{
    border: 1px solid {BORDER}; border-left-width: 4px; border-radius: 10px;
    padding: .6rem 1rem; margin-bottom: .5rem; background: rgba(128,128,128,.06);
}}
</style>
"""


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", eyebrow: str = "") -> None:
    eb = f'<div class="fi-eyebrow">{eyebrow}</div>' if eyebrow else ""
    sub = f'<p class="fi-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="fi-header">{eb}<div class="fi-title">{title}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def card(title: str, text: str = "", step: str = "", center: bool = False) -> None:
    st_html = f'<div class="fi-step">{step}</div><br>' if step else ""
    cls = "fi-card center" if center else "fi-card"
    st.markdown(
        f'<div class="{cls}">{st_html}<div class="fi-card-title">{title}</div>'
        f'<p class="fi-card-text">{text}</p></div>',
        unsafe_allow_html=True,
    )


def result_card(title: str, caption: str, color: str) -> None:
    st.markdown(
        f'<div class="fi-result" style="border-left-color:{color};">'
        f'<h2 style="color:{color};">{title}</h2>'
        f'<p class="fi-muted" style="margin:.2rem 0 0;">{caption}</p></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int | None = None):
    """Make a plotly figure match the light theme."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#9aa4b2"),
        margin=dict(t=40, b=20, l=10, r=10),
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,.18)", zerolinecolor="rgba(128,128,128,.3)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zerolinecolor="rgba(128,128,128,.3)")
    if height:
        fig.update_layout(height=height)
    return fig
