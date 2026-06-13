from sentiment.finbert_model import FinBERTAnalyzer
from sentiment.news_fetcher import NewsFetcher
from utils.logger import get_logger

logger = get_logger(__name__)


class SentimentPipeline:

    def __init__(self):
        self.analyzer = FinBERTAnalyzer()
        self.fetcher  = NewsFetcher()

    def analyze_company_news(self, company: str) -> dict:
        logger.info(f"Running sentiment pipeline for: {company}")

        articles = self.fetcher.fetch_news(company)
        if not articles:
            return {"error": f"No news found for {company}"}

        texts = [
            a["title"] + ". " + (a["description"] or "")
            for a in articles
        ]

        results = self.analyzer.analyze_batch(texts)

        for i, article in enumerate(articles):
            article["sentiment"]  = results[i]["sentiment"]
            article["confidence"] = results[i]["confidence"]
            article["scores"]     = results[i]["scores"]

        counts = {"positive": 0, "negative": 0, "neutral": 0}
        for r in results:
            counts[r["sentiment"]] += 1

        total   = len(results)
        overall = max(counts, key=counts.get)

        sentiment_score = round(
            (counts["positive"] - counts["negative"]) / total * 100, 1
        )

        return {
            "company":          company,
            "overall_sentiment": overall,
            "sentiment_score":  sentiment_score,
            "distribution": {
                k: round(v / total * 100, 1)
                for k, v in counts.items()
            },
            "total_articles":   total,
            "articles":         articles
        }

    def analyze_multiple_companies(self, companies: list[str]) -> dict:
        return {
            company: self.analyze_company_news(company)
            for company in companies
        }

    def analyze_custom_text(self, text: str) -> dict:
        return self.analyzer.analyze(text)