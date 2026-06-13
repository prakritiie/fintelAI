import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from config.settings import APP_TITLE, APP_ICON

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    .main-header {
        background: linear-gradient(135deg, #0f172a, #1e3a5f);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .status-ok {
        color: #22c55e;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="color:#38bdf8; font-size:2.5rem; margin:0;">📊 FinIntel AI</h1>
    <p style="color:#94a3b8; font-size:1.1rem; margin-top:0.5rem;">
        Multi-Agent Financial Intelligence & Risk Analysis Platform
    </p>
</div>
""", unsafe_allow_html=True)

# ── Overview Cards ───────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size:2rem;">📄</div>
        <div style="color:#38bdf8; font-weight:bold;">Document RAG</div>
        <div style="color:#94a3b8; font-size:0.85rem;">Upload & query financial PDFs</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size:2rem;">🧠</div>
        <div style="color:#a78bfa; font-weight:bold;">Sentiment Analysis</div>
        <div style="color:#94a3b8; font-size:0.85rem;">FinBERT-powered news analysis</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size:2rem;">⚠️</div>
        <div style="color:#f59e0b; font-weight:bold;">Risk Prediction</div>
        <div style="color:#94a3b8; font-size:0.85rem;">ML-based financial risk scoring</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <div style="font-size:2rem;">🤖</div>
        <div style="color:#34d399; font-weight:bold;">AI Agents</div>
        <div style="color:#94a3b8; font-size:0.85rem;">LangGraph multi-agent workflows</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Navigation Guide ─────────────────────────────────────────
st.markdown("### 🗺️ Navigate Using the Sidebar")

col1, col2 = st.columns(2)

with col1:
    st.info("📁 **Upload Documents** — Upload 10-K, annual reports, earnings PDFs")
    st.info("💬 **AI Chat** — Ask questions about uploaded documents")
    st.info("📰 **Sentiment Dashboard** — Analyze financial news sentiment")

with col2:
    st.warning("📊 **Risk Dashboard** — Predict company financial risk")
    st.warning("🕸️ **Knowledge Graph** — Visualize company relationships")
    st.warning("🤖 **Agent Workflow** — Watch multi-agent AI in action")

# ── System Status ────────────────────────────────────────────
st.markdown("### 🔧 System Status")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    try:
        from core.vector_store import VectorStore
        vs = VectorStore()
        count = vs.get_collection_count()
        st.success(f"✅ Vector DB — {count} chunks indexed")
    except Exception as e:
        st.error(f"❌ Vector DB — {e}")

with status_col2:
    try:
        from config.settings import GROQ_API_KEY
        if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
            st.success("✅ Groq LLM — Connected")
        else:
            st.warning("⚠️ Groq LLM — API key not set")
    except Exception as e:
        st.error(f"❌ Groq LLM — {e}")

with status_col3:
    try:
        from config.settings import NEWS_API_KEY
        if NEWS_API_KEY and NEWS_API_KEY != "your_news_api_key_here":
            st.success("✅ News API — Connected")
        else:
            st.warning("⚠️ News API — Not configured (optional)")
    except Exception as e:
        st.error(f"❌ News API — {e}")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#475569;'>FinIntel AI — Built with LangChain · LangGraph · FinBERT · ChromaDB · Streamlit</p>",
    unsafe_allow_html=True
)