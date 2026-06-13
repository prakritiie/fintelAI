import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import time

st.set_page_config(page_title="Agent Workflow", page_icon="🤖", layout="wide")

st.markdown("## 🤖 Multi-Agent AI Workflow")
st.markdown("Watch **5 specialized AI agents** collaborate to analyze a company end-to-end.")

# ── Agent descriptions ───────────────────────────────────────
st.markdown("### 🧩 Agent Pipeline")
cols = st.columns(5)
agents_info = [
    ("📄", "Document Agent",     "Extracts & analyzes financial report sections"),
    ("⚠️", "Risk Agent",         "Predicts financial risk using XGBoost"),
    ("📰", "News Agent",         "Analyzes market sentiment via FinBERT"),
    ("📝", "Summary Agent",      "Generates executive summary with LLM"),
    ("💡", "Recommendation Agent","Produces investment signal & recommendation"),
]
for i, (icon, name, desc) in enumerate(agents_info):
    with cols[i]:
        st.markdown(f"""
        <div style='background:#1e293b; border-radius:10px; padding:1rem; text-align:center; height:140px;'>
            <div style='font-size:2rem;'>{icon}</div>
            <div style='color:#38bdf8; font-weight:bold; font-size:0.85rem;'>{name}</div>
            <div style='color:#94a3b8; font-size:0.75rem; margin-top:0.3rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Inputs ───────────────────────────────────────────────────
st.markdown("### ⚙️ Configure Workflow")
col1, col2 = st.columns(2)

with col1:
    company_name = st.text_input("Company Name", value="Apple")
    doc_name     = st.text_input("Document Name", value="Apple 10-K 2023")

with col2:
    st.markdown("**Financial Metrics (for Risk Agent)**")
    fc1, fc2 = st.columns(2)
    with fc1:
        roa           = st.number_input("ROA",            value=0.19, step=0.01, format="%.2f")
        current_ratio = st.number_input("Current Ratio",  value=1.80, step=0.10, format="%.2f")
        profit_margin = st.number_input("Profit Margin",  value=0.25, step=0.01, format="%.2f")
        interest_cov  = st.number_input("Interest Coverage", value=18.0, step=0.5, format="%.1f")
    with fc2:
        roe           = st.number_input("ROE",            value=0.35, step=0.01, format="%.2f")
        debt_equity   = st.number_input("Debt/Equity",    value=1.20, step=0.10, format="%.2f")
        rev_growth    = st.number_input("Revenue Growth", value=0.08, step=0.01, format="%.2f")
        cash_ratio    = st.number_input("Cash Ratio",     value=0.90, step=0.10, format="%.2f")

run_btn = st.button("🚀 Run Multi-Agent Workflow", type="primary", use_container_width=True)

# ── Run workflow ─────────────────────────────────────────────
if run_btn:
    financials = {
        "roa": roa, "roe": roe,
        "current_ratio": current_ratio,
        "debt_to_equity": debt_equity,
        "profit_margin": profit_margin,
        "revenue_growth": rev_growth,
        "interest_coverage": interest_cov,
        "cash_ratio": cash_ratio
    }

    st.markdown("### 🔄 Agent Execution")
    progress_bar  = st.progress(0, text="Initializing workflow...")
    status_boxes  = st.empty()
    log_container = st.empty()

    completed_logs = []

    def progress_callback(step_idx, agent_name, agent_icon):
        pct = int((step_idx / 5) * 100)
        progress_bar.progress(pct, text=f"Running: {agent_icon} {agent_name}...")
        completed_logs.append(f"⏳ {agent_icon} {agent_name} — running...")
        log_container.markdown("\n\n".join(completed_logs))
        time.sleep(0.3)

    try:
        from agents.workflow import FinIntelWorkflow
        workflow = FinIntelWorkflow()
        state    = workflow.run(
            company_name=company_name,
            doc_name=doc_name,
            financials=financials,
            progress_callback=progress_callback
        )

        progress_bar.progress(100, text="✅ Workflow complete!")

        # Agent logs
        st.markdown("### 📋 Agent Execution Log")
        for log in state["agent_logs"]:
            st.success(log)

        st.divider()

        # Results tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📝 Executive Summary",
            "⚠️ Risk Result",
            "📰 Sentiment",
            "💡 Recommendation"
        ])

        with tab1:
            st.markdown(state.get("executive_summary", "No summary generated."))

        with tab2:
            risk = state.get("risk_result", {})
            if risk:
                st.markdown(f"""
                <div style='background:#1e293b; border:2px solid {risk.get("risk_color","#gray")};
                            border-radius:12px; padding:1.5rem; text-align:center;'>
                    <h2 style='color:{risk.get("risk_color","#fff")};'>
                        {risk.get("risk_icon","")} {risk.get("risk_label","")}
                    </h2>
                    <p style='color:#94a3b8;'>Confidence: {risk.get("confidence","")}%</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                probs = risk.get("probabilities", {})
                if probs:
                    import plotly.graph_objects as go
                    fig = go.Figure(go.Bar(
                        x=list(probs.values()),
                        y=list(probs.keys()),
                        orientation="h",
                        marker_color=["#22c55e", "#f59e0b", "#ef4444"]
                    ))
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0", height=250
                    )
                    st.plotly_chart(fig, use_container_width=True)

        with tab3:
            sent = state.get("sentiment_result", {})
            if sent:
                s_col1, s_col2, s_col3 = st.columns(3)
                s_col1.metric("Overall Sentiment", sent.get("overall_sentiment","").capitalize())
                s_col2.metric("Sentiment Score",   f"{sent.get('sentiment_score', 0):+.1f}")
                s_col3.metric("Articles Analyzed", sent.get("total_articles", 0))

                dist = sent.get("distribution", {})
                if dist:
                    import plotly.graph_objects as go
                    fig2 = go.Figure(go.Pie(
                        labels=list(dist.keys()),
                        values=list(dist.values()),
                        hole=0.4,
                        marker_colors=["#22c55e", "#ef4444", "#f59e0b"]
                    ))
                    fig2.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0", height=300
                    )
                    st.plotly_chart(fig2, use_container_width=True)

        with tab4:
            signal = state.get("signal", "")
            if signal:
                signal_color = "#22c55e" if "BULL" in signal else "#ef4444" if "BEAR" in signal else "#f59e0b"
                st.markdown(f"""
                <div style='background:#1e293b; border:2px solid {signal_color};
                            border-radius:12px; padding:1rem; text-align:center; margin-bottom:1rem;'>
                    <h2 style='color:{signal_color};'>{signal}</h2>
                </div>
                """, unsafe_allow_html=True)
            st.markdown(state.get("recommendation", "No recommendation generated."))

    except Exception as e:
        st.error(f"Workflow error: {e}")
        st.exception(e)