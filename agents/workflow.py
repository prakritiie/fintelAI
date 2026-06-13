from agents.document_agent      import DocumentAgent
from agents.risk_agent          import RiskAgent
from agents.news_agent          import NewsAgent
from agents.summary_agent       import SummaryAgent
from agents.recommendation_agent import RecommendationAgent
from utils.logger import get_logger

logger = get_logger(__name__)


def build_initial_state(company_name: str, doc_name: str, financials: dict = None) -> dict:
    return {
        "company_name":      company_name,
        "doc_name":          doc_name,
        "financials":        financials or {},
        "document_analysis": "",
        "sections":          {},
        "risk_result":       {},
        "sentiment_result":  {},
        "executive_summary": "",
        "recommendation":    "",
        "signal":            "",
        "agent_logs":        []
    }


class FinIntelWorkflow:

    def __init__(self):
        self.document_agent      = DocumentAgent()
        self.risk_agent          = RiskAgent()
        self.news_agent          = NewsAgent()
        self.summary_agent       = SummaryAgent()
        self.recommendation_agent = RecommendationAgent()

        self.steps = [
            self.document_agent,
            self.risk_agent,
            self.news_agent,
            self.summary_agent,
            self.recommendation_agent
        ]

    def run(self, company_name: str, doc_name: str, financials: dict = None,
            progress_callback=None) -> dict:

        logger.info(f"Starting FinIntel workflow for: {company_name}")
        state = build_initial_state(company_name, doc_name, financials)

        for i, agent in enumerate(self.steps):
            logger.info(f"Running agent {i+1}/{len(self.steps)}: {agent.name}")

            if progress_callback:
                progress_callback(i, agent.name, agent.icon)

            state = agent.run(state)

        logger.info("Workflow completed successfully")
        return state