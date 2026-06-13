from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from config.settings import GROQ_API_KEY, GROQ_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

RISK_SCORE_MAP = {"Low Risk": 1, "Medium Risk": 2, "High Risk": 3}


class RecommendationAgent:
    name = "Recommendation Agent"
    icon = "💡"

    def __init__(self):
        self.llm    = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.2)
        self.parser = StrOutputParser()

    def run(self, state: dict) -> dict:
        logger.info("RecommendationAgent running...")

        risk_result  = state.get("risk_result",  {})
        sentiment    = state.get("sentiment_result", {})
        summary      = state.get("executive_summary", "")
        company      = state.get("company_name", "the company")

        risk_label   = risk_result.get("risk_label",  "Medium Risk")
        risk_score   = RISK_SCORE_MAP.get(risk_label, 2)
        sentiment_ov = sentiment.get("overall_sentiment", "neutral")
        sent_score   = sentiment.get("sentiment_score", 0)

        # Rule-based signal
        if risk_score == 1 and sentiment_ov == "positive":
            signal = "BULLISH 📈"
        elif risk_score == 3 or sentiment_ov == "negative":
            signal = "BEARISH 📉"
        else:
            signal = "NEUTRAL ⚖️"

        prompt = f"""
You are a top investment research analyst.

Company: {company}
Risk Level: {risk_label}
Market Sentiment: {sentiment_ov.capitalize()} (score: {sent_score:+.1f})
Overall Signal: {signal}

Executive Summary:
{summary[:1500]}

Provide a structured investment recommendation with:
1. **Overall Signal**: {signal}
2. **Key Strengths** (2-3 bullet points)
3. **Key Risks** (2-3 bullet points)
4. **Recommendation**: One clear actionable statement

Be concise, professional, and data-driven.
"""
        try:
            response     = self.llm.invoke(prompt)
            rec_text     = self.parser.invoke(response)

            state["recommendation"] = rec_text
            state["signal"]         = signal
            state["agent_logs"].append(
                f"{self.icon} {self.name}: Signal — {signal}"
            )

        except Exception as e:
            logger.error(f"RecommendationAgent error: {e}")
            state["recommendation"] = "Recommendation generation failed."
            state["agent_logs"].append(f"{self.icon} {self.name}: Error — {e}")

        return state