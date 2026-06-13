from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from config.settings import GROQ_API_KEY, GROQ_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)


class SummaryAgent:
    name = "Summary Agent"
    icon = "📝"

    def __init__(self):
        self.llm    = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0.1)
        self.parser = StrOutputParser()

    def run(self, state: dict) -> dict:
        logger.info("SummaryAgent running...")

        doc_analysis = state.get("document_analysis", "No document analysis available.")
        risk_result  = state.get("risk_result",  {})
        sentiment    = state.get("sentiment_result", {})
        company      = state.get("company_name", "the company")

        prompt = f"""
You are a senior financial analyst. Synthesize the following data into a concise executive summary.

Company: {company}

Document Analysis:
{doc_analysis[:2000]}

Risk Assessment:
- Risk Level: {risk_result.get("risk_label", "N/A")}
- Confidence: {risk_result.get("confidence", "N/A")}%

Market Sentiment:
- Overall: {sentiment.get("overall_sentiment", "N/A")}
- Sentiment Score: {sentiment.get("sentiment_score", "N/A")}
- Articles Analyzed: {sentiment.get("total_articles", "N/A")}

Write a 3-paragraph executive summary covering:
1. Business overview and financial performance
2. Risk assessment and key concerns
3. Market sentiment and outlook

Keep it professional and data-driven.
"""
        try:
            summary = self.llm.invoke(prompt)
            summary_text = self.parser.invoke(summary)

            state["executive_summary"] = summary_text
            state["agent_logs"].append(
                f"{self.icon} {self.name}: Executive summary generated ({len(summary_text)} chars)"
            )

        except Exception as e:
            logger.error(f"SummaryAgent error: {e}")
            state["executive_summary"] = "Summary generation failed."
            state["agent_logs"].append(f"{self.icon} {self.name}: Error — {e}")

        return state