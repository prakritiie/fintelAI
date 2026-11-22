import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dashboard.ui import apply_theme, page_header, style_fig, result_card, MUTED, GOOD, WARN, BAD

st.set_page_config(page_title="Risk Dashboard", page_icon="⚠️", layout="wide")

apply_theme()
page_header("Risk Prediction", "Estimate financial risk with XGBoost and see exactly which metrics drive the score using SHAP.", eyebrow="Step 4")

# ── Load predictor ───────────────────────────────────────────
@st.cache_resource
def load_predictor():
    from risk.predictor import RiskPredictor
    return RiskPredictor()

with st.spinner("Loading risk model..."):
    predictor = load_predictor()
st.caption("Risk model ready")

# ── Sidebar — manual input ───────────────────────────────────
with st.sidebar:
    st.markdown("### Financial metrics")
    roa               = st.slider("Return on Assets (ROA)",        -0.30, 0.30,  0.10, 0.01)
    roe               = st.slider("Return on Equity (ROE)",        -0.50, 0.50,  0.15, 0.01)
    current_ratio     = st.slider("Current Ratio",                  0.10, 5.00,  1.50, 0.10)
    debt_to_equity    = st.slider("Debt to Equity",                 0.00, 7.00,  1.00, 0.10)
    profit_margin     = st.slider("Profit Margin",                 -0.30, 0.40,  0.10, 0.01)
    revenue_growth    = st.slider("Revenue Growth",                -0.30, 0.40,  0.05, 0.01)
    interest_coverage = st.slider("Interest Coverage Ratio",        0.00, 25.00, 5.00, 0.50)
    cash_ratio        = st.slider("Cash Ratio",                     0.00, 3.00,  0.50, 0.10)
    predict_btn = st.button("Predict risk", type="primary", use_container_width=True)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Custom prediction", "Sample companies"])

# ── Tab 1: Custom prediction ─────────────────────────────────
with tab1:
    if predict_btn:
        financials = {
            "roa": roa, "roe": roe,
            "current_ratio": current_ratio,
            "debt_to_equity": debt_to_equity,
            "profit_margin": profit_margin,
            "revenue_growth": revenue_growth,
            "interest_coverage": interest_coverage,
            "cash_ratio": cash_ratio
        }

        result = predictor.predict(financials)

        # Risk header
        result_card(f'{result["risk_icon"]} {result["risk_label"]}',
                    f'Confidence: {result["confidence"]}%', result["risk_color"])

        # Probability gauge
        col1, col2 = st.columns([1, 1])
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result["probabilities"].get("High Risk", 0),
                title={"text": "High risk probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar":  {"color": result["risk_color"]},
                    "steps": [
                        {"range": [0,  33], "color": "rgba(22,163,74,.22)"},
                        {"range": [33, 66], "color": "rgba(217,119,6,.22)"},
                        {"range": [66, 100],"color": "rgba(220,38,38,.22)"},
                    ]
                }
            ))
            style_fig(fig, 300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            probs = result["probabilities"]
            fig2 = go.Figure(go.Bar(
                x=list(probs.values()),
                y=list(probs.keys()),
                orientation="h",
                marker_color=[GOOD, WARN, BAD]
            ))
            fig2.update_layout(title="Risk probabilities (%)")
            style_fig(fig2, 300)
            st.plotly_chart(fig2, use_container_width=True)

        # SHAP explanation
        st.markdown("### Why this score? (SHAP)")
        try:
            from risk.explainer import RiskExplainer
            explainer = RiskExplainer(predictor.model, predictor.scaler)
            explanation = explainer.explain(financials)

            rows = []
            for f in explanation["all_factors"]:
                rows.append({
                    "Feature":    f["label"],
                    "Value":      f["value"],
                    "SHAP Impact": f["shap_value"],
                    "Effect":     f["impact"]
                })

            df = pd.DataFrame(rows)
            fig3 = px.bar(
                df, x="SHAP Impact", y="Feature",
                orientation="h",
                color="SHAP Impact",
                color_continuous_scale=[GOOD, WARN, BAD],
                title="Feature impact on risk score"
            )
            style_fig(fig3, 400)
            st.plotly_chart(fig3, use_container_width=True)

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")

    else:
        st.info("Adjust the financial metrics in the sidebar, then click **Predict risk**.")

# ── Tab 2: Sample companies ───────────────────────────────────
with tab2:
    st.markdown("### Sample companies")
    samples = predictor.predict_sample_companies()

    cols = st.columns(len(samples))
    for i, result in enumerate(samples):
        with cols[i]:
            st.markdown(f"""
            <div class='fi-card center' style='border-top:4px solid {result["risk_color"]};'>
                <div class='fi-card-title' style='color:{MUTED};'>{result["company_name"]}</div>
                <h3 style='color:{result["risk_color"]}; margin:.3rem 0;'>
                    {result["risk_icon"]} {result["risk_label"]}
                </h3>
                <p class='fi-card-text'>Confidence {result["confidence"]}%</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison table
    rows = []
    for result in samples:
        row = {"Company": result["company_name"], "Risk": result["risk_label"]}
        row.update(result["probabilities"])
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)