import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from config.settings import APP_TITLE, APP_ICON
from dashboard.ui import apply_theme, page_header, card

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)
apply_theme()

# ── Header ───────────────────────────────────────────────────
page_header(
    "FinIntel AI",
    "Multi-agent financial intelligence and risk analysis. Upload a filing, ask questions, "
    "gauge sentiment and risk, and get an investment view in minutes.",
    eyebrow="Financial Intelligence Platform",
)

# ── Capabilities ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    card("Document RAG", "Upload 10-Ks and reports, then ask questions with cited sources.")
with c2:
    card("Sentiment Analysis", "FinBERT scores the tone of the latest financial news.")
with c3:
    card("Risk Prediction", "XGBoost risk scoring with SHAP explanations for every factor.")
with c4:
    card("AI Agents", "Five LangGraph agents collaborate on a full company analysis.")

# ── Getting started ──────────────────────────────────────────
st.markdown("### Get started")
st.caption("Use the sidebar to open any of these tools.")

r1 = st.columns(3)
r2 = st.columns(3)
steps = [
    ("1", "Upload Documents", "Index 10-K filings, annual reports and earnings PDFs."),
    ("2", "AI Chat", "Ask questions about your indexed documents."),
    ("3", "Sentiment", "Analyze the tone of recent news for a company."),
    ("4", "Risk", "Predict financial risk and see what drives it."),
    ("5", "Knowledge Graph", "Explore companies, competitors, products and risks."),
    ("6", "Agent Workflow", "Run all agents end-to-end for an investment signal."),
]
for col, (n, title, text) in zip(r1 + r2, steps):
    with col:
        card(title, text, step=n)
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

# ── System Status ────────────────────────────────────────────
st.markdown("### System status")

s1, s2, s3 = st.columns(3)

with s1:
    try:
        from core.vector_store import VectorStore
        count = VectorStore().get_collection_count()
        st.success(f"Vector DB: {count} chunks indexed")
    except Exception as e:
        st.error(f"Vector DB: {e}")

with s2:
    try:
        from config.settings import GROQ_API_KEY
        if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
            st.success("Groq LLM: API key configured")
        else:
            st.warning("Groq LLM: API key not set")
    except Exception as e:
        st.error(f"Groq LLM: {e}")

with s3:
    try:
        from config.settings import NEWS_API_KEY
        if NEWS_API_KEY and NEWS_API_KEY != "your_news_api_key_here":
            st.success("News API: API key configured")
        else:
            st.warning("News API: not configured (optional)")
    except Exception as e:
        st.error(f"News API: {e}")

st.markdown("---")
st.caption("Built with LangChain · LangGraph · FinBERT · ChromaDB · Streamlit")
