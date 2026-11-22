import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dashboard.ui import apply_theme, page_header, style_fig, SENTIMENT_COLORS, MUTED, TEXT

st.set_page_config(page_title="Sentiment Dashboard", page_icon="📰", layout="wide")

apply_theme()
page_header("Sentiment Analysis", "Score the tone of recent financial news with FinBERT, a finance-specific language model.", eyebrow="Step 3")

# ── Load pipeline (cached) ───────────────────────────────────
@st.cache_resource
def load_pipeline():
    from sentiment.sentiment_pipeline import SentimentPipeline
    return SentimentPipeline()

with st.spinner("Loading FinBERT model (first time may take ~30 seconds)..."):
    pipeline = load_pipeline()

st.caption("FinBERT model ready")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    companies = st.multiselect(
        "Select Companies",
        ["Apple", "Tesla", "Microsoft", "General"],
        default=["Apple", "Tesla"]
    )
    st.markdown("### Custom text analysis")
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

    st.markdown("### Custom text result")
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

if st.button("Run sentiment analysis", type="primary"):
    st.session_state.sentiment_results = {}

    for company in companies:
        with st.spinner(f"Analyzing {company} news..."):
            result = pipeline.analyze_company_news(company)
            st.session_state.sentiment_results[company] = result

if "sentiment_results" not in st.session_state:
    st.info("Select companies in the sidebar, then click **Run sentiment analysis**.")
    st.stop()

results = st.session_state.sentiment_results

# ── Overview metrics ─────────────────────────────────────────
st.markdown("### Sentiment overview")
cols = st.columns(len(results))

for i, (company, data) in enumerate(results.items()):
    with cols[i]:
        sentiment = data["overall_sentiment"]
        score     = data["sentiment_score"]
        color_map = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
        st.markdown(f"""
        <div class='fi-card center'>
            <div class='fi-card-title' style='color:{MUTED};'>{company}</div>
            <h2 style='margin:.3rem 0; color:{SENTIMENT_COLORS[sentiment]};'>{sentiment.capitalize()}</h2>
            <p class='fi-card-text'>Score {score:+.1f} · {data['total_articles']} articles</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Distribution charts ──────────────────────────────────────
st.markdown("### Sentiment distribution")
chart_cols = st.columns(len(results))

for i, (company, data) in enumerate(results.items()):
    with chart_cols[i]:
        dist = data["distribution"]
        fig = go.Figure(go.Pie(
            labels=list(dist.keys()),
            values=list(dist.values()),
            hole=0.4,
            marker_colors=[SENTIMENT_COLORS[k] for k in dist.keys()]
        ))
        fig.update_layout(title=company)
        style_fig(fig, 300)
        st.plotly_chart(fig, use_container_width=True)

# ── Comparison bar chart ─────────────────────────────────────
if len(results) > 1:
    st.markdown("### Company comparison")
    rows = []
    for company, data in results.items():
        for sentiment, pct in data["distribution"].items():
            rows.append({"Company": company, "Sentiment": sentiment, "Percentage": pct})

    df = pd.DataFrame(rows)
    fig = px.bar(
        df, x="Company", y="Percentage", color="Sentiment",
        barmode="group",
        color_discrete_map=SENTIMENT_COLORS
    )
    style_fig(fig, 400)
    st.plotly_chart(fig, use_container_width=True)

# ── News articles table ──────────────────────────────────────
st.markdown("### Article-level sentiment")
for company, data in results.items():
    with st.expander(f"{company}: {data['total_articles']} articles", expanded=False):
        for article in data["articles"]:
            sentiment = article["sentiment"]
            color_map = SENTIMENT_COLORS
            st.markdown(f"""
            <div class='fi-article' style='border-left-color:{color_map[sentiment]};'>
                <strong style='color:{TEXT};'>{article['title']}</strong><br>
                <span style='color:{color_map[sentiment]}; font-weight:600;'>
                    {sentiment.capitalize()} ({article['confidence']*100:.1f}%)
                </span>
                <span style='color:{MUTED}; float:right;'>{article['source']} · {article['published']}</span>
            </div>
            """, unsafe_allow_html=True)