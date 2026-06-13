import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Sentiment Dashboard", page_icon="📰", layout="wide")

st.markdown("## 📰 Financial Sentiment Analysis")
st.markdown("Analyze market sentiment on financial news using **FinBERT** — a financial domain NLP model.")

# ── Load pipeline (cached) ───────────────────────────────────
@st.cache_resource
def load_pipeline():
    from sentiment.sentiment_pipeline import SentimentPipeline
    return SentimentPipeline()

with st.spinner("Loading FinBERT model (first time may take ~30 seconds)..."):
    pipeline = load_pipeline()

st.success("✅ FinBERT model ready")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    companies = st.multiselect(
        "Select Companies",
        ["Apple", "Tesla", "Microsoft", "General"],
        default=["Apple", "Tesla"]
    )
    st.markdown("### 📝 Custom Text Analysis")
    custom_text = st.text_area(
        "Enter any financial text:",
        placeholder="e.g. Company reports record revenue growth..."
    )
    analyze_custom_btn = st.button("Analyze Text", use_container_width=True)

# ── Custom text analysis ─────────────────────────────────────
if analyze_custom_btn and custom_text:
    result = pipeline.analyze_custom_text(custom_text)
    sentiment = result["sentiment"]
    color = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}

    st.markdown("### 🔍 Custom Text Result")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sentiment",  f"{color[sentiment]} {sentiment.capitalize()}")
    col2.metric("Positive",   f"{result['scores']['positive']*100:.1f}%")
    col3.metric("Negative",   f"{result['scores']['negative']*100:.1f}%")
    col4.metric("Neutral",    f"{result['scores']['neutral']*100:.1f}%")
    st.divider()

# ── Company analysis ─────────────────────────────────────────
if not companies:
    st.warning("Please select at least one company from the sidebar.")
    st.stop()

if st.button("🚀 Run Sentiment Analysis", type="primary"):
    st.session_state.sentiment_results = {}

    for company in companies:
        with st.spinner(f"Analyzing {company} news..."):
            result = pipeline.analyze_company_news(company)
            st.session_state.sentiment_results[company] = result

if "sentiment_results" not in st.session_state:
    st.info("👈 Select companies and click **Run Sentiment Analysis**")
    st.stop()

results = st.session_state.sentiment_results

# ── Overview metrics ─────────────────────────────────────────
st.markdown("### 📊 Sentiment Overview")
cols = st.columns(len(results))

for i, (company, data) in enumerate(results.items()):
    with cols[i]:
        sentiment = data["overall_sentiment"]
        score     = data["sentiment_score"]
        color_map = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
        st.markdown(f"""
        <div style='background:#1e293b; border-radius:10px; padding:1rem; text-align:center;'>
            <h3 style='color:#38bdf8;'>{company}</h3>
            <h2>{color_map[sentiment]} {sentiment.capitalize()}</h2>
            <p style='color:#94a3b8;'>Score: {score:+.1f}</p>
            <p style='color:#94a3b8;'>{data['total_articles']} articles analyzed</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Distribution charts ──────────────────────────────────────
st.markdown("### 🥧 Sentiment Distribution")
chart_cols = st.columns(len(results))

for i, (company, data) in enumerate(results.items()):
    with chart_cols[i]:
        dist = data["distribution"]
        fig = go.Figure(go.Pie(
            labels=list(dist.keys()),
            values=list(dist.values()),
            hole=0.4,
            marker_colors=["#22c55e", "#ef4444", "#f59e0b"]
        ))
        fig.update_layout(
            title=company,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            height=300,
            margin=dict(t=40, b=0, l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Comparison bar chart ─────────────────────────────────────
if len(results) > 1:
    st.markdown("### 📊 Company Comparison")
    rows = []
    for company, data in results.items():
        for sentiment, pct in data["distribution"].items():
            rows.append({"Company": company, "Sentiment": sentiment, "Percentage": pct})

    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="Company", y="Percentage", color="Sentiment",
        barmode="group",
        color_discrete_map={
            "positive": "#22c55e",
            "negative": "#ef4444",
            "neutral":  "#f59e0b"
        }
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ── News articles table ──────────────────────────────────────
st.markdown("### 📰 Article-Level Sentiment")
for company, data in results.items():
    with st.expander(f"📰 {company} — {data['total_articles']} Articles", expanded=False):
        for article in data["articles"]:
            sentiment = article["sentiment"]
            color_map = {"positive": "#22c55e", "negative": "#ef4444", "neutral": "#f59e0b"}
            st.markdown(f"""
            <div style='border-left: 4px solid {color_map[sentiment]};
                        padding: 0.5rem 1rem; margin-bottom:0.5rem;
                        background:#1e293b; border-radius:0 8px 8px 0;'>
                <strong style='color:#e2e8f0;'>{article['title']}</strong><br>
                <span style='color:{color_map[sentiment]};'>
                    {sentiment.capitalize()} ({article['confidence']*100:.1f}%)
                </span>
                <span style='color:#64748b; float:right;'>{article['source']} · {article['published']}</span>
            </div>
            """, unsafe_allow_html=True)