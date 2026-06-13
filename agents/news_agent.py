from sentiment.sentiment_pipeline import SentimentPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


class NewsAgent:
    name = "News Intelligence Agent"
    icon = "📰"

    def __init__(self):
        self.pipeline = SentimentPipeline()

    def run(self, state: dict) -> dict:
        logger.info("NewsAgent running...")

        company = state.get("company_name", "General")

        try:
            result = self.pipeline.analyze_company_news(company)

            state["sentiment_result"] = result
            state["agent_logs"].append(
                f"{self.icon} {self.name}: Analyzed {result['total_articles']} articles "
                f"for {company} — Overall: {result['overall_sentiment'].capitalize()} "
                f"(score: {result['sentiment_score']:+.1f})"
            )

        except Exception as e:
            logger.error(f"NewsAgent error: {e}")
            state["sentiment_result"] = {"overall_sentiment": "neutral", "sentiment_score": 0}
            state["agent_logs"].append(f"{self.icon} {self.name}: Error — {e}")

        return state