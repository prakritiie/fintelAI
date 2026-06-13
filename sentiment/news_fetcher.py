import requests
from datetime import datetime, timedelta
from config.settings import NEWS_API_KEY
from utils.logger import get_logger

logger = get_logger(__name__)

# Fallback headlines when no API key is set
SAMPLE_HEADLINES = {
    "Apple": [
        "Apple reports record Services revenue in Q4 2023",
        "Apple faces antitrust scrutiny over App Store policies",
        "iPhone 15 sales disappoint analysts in China market",
        "Apple expands manufacturing presence in India",
        "Apple stock hits all-time high after strong earnings",
        "Supply chain concerns weigh on Apple's outlook",
        "Apple Vision Pro launch generates mixed reviews",
        "Apple increases dividend and share buyback program",
    ],
    "Tesla": [
        "Tesla cuts prices again amid growing EV competition",
        "Tesla reports strong delivery numbers for Q3",
        "Tesla faces regulatory investigation over autopilot",
        "Tesla expands Gigafactory capacity in Texas",
        "Tesla stock surges on better than expected margins",
        "Tesla faces increased competition from Chinese EV makers",
        "Tesla energy storage business hits record revenue",
        "Elon Musk sells Tesla shares raising investor concerns",
    ],
    "Microsoft": [
        "Microsoft Azure cloud revenue grows 28 percent",
        "Microsoft Copilot AI integration drives Office 365 growth",
        "Microsoft completes Activision Blizzard acquisition",
        "Microsoft faces EU regulatory concerns over AI products",
        "Microsoft stock reaches new highs on AI optimism",
        "Microsoft LinkedIn revenue growth slows down",
        "Microsoft invests additional 10 billion in OpenAI",
        "Microsoft gaming division struggles post acquisition",
    ],
    "General": [
        "Federal Reserve signals potential rate cuts in 2024",
        "US inflation data shows continued cooling trend",
        "Global markets rally on positive economic data",
        "Tech sector leads market gains this quarter",
        "Banking sector faces headwinds from high interest rates",
        "Oil prices rise amid Middle East tensions",
        "Strong jobs report eases recession fears",
        "Dollar weakens as Fed pivot expectations grow",
    ]
}


class NewsFetcher:

    def __init__(self):
        self.api_key = NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch_news(self, company: str, days_back: int = 7, max_articles: int = 10) -> list[dict]:
        if not self.api_key or self.api_key == "your_news_api_key_here":
            logger.warning("No NewsAPI key — using sample headlines")
            return self._get_sample_news(company)

        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            params = {
                "q":        f"{company} stock finance earnings",
                "from":     from_date,
                "language": "en",
                "sortBy":   "relevancy",
                "pageSize": max_articles,
                "apiKey":   self.api_key
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            articles = response.json().get("articles", [])

            return [
                {
                    "title":       a.get("title", ""),
                    "description": a.get("description", ""),
                    "source":      a.get("source", {}).get("name", "Unknown"),
                    "published":   a.get("publishedAt", "")[:10],
                    "url":         a.get("url", "")
                }
                for a in articles if a.get("title")
            ]

        except Exception as e:
            logger.error(f"NewsAPI error: {e} — using sample data")
            return self._get_sample_news(company)

    def _get_sample_news(self, company: str) -> list[dict]:
        headlines = SAMPLE_HEADLINES.get(company, SAMPLE_HEADLINES["General"])
        return [
            {
                "title":       h,
                "description": h,
                "source":      "Sample Data",
                "published":   datetime.now().strftime("%Y-%m-%d"),
                "url":         ""
            }
            for h in headlines
        ]

    def get_multiple_companies(self, companies: list[str]) -> dict:
        return {company: self.fetch_news(company) for company in companies}