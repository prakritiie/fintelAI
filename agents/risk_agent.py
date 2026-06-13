from risk.predictor import RiskPredictor
from utils.logger import get_logger

logger = get_logger(__name__)

# Default financials when none provided
DEFAULT_FINANCIALS = {
    "roa":               0.08,
    "roe":               0.12,
    "current_ratio":     1.50,
    "debt_to_equity":    1.20,
    "profit_margin":     0.10,
    "revenue_growth":    0.05,
    "interest_coverage": 5.00,
    "cash_ratio":        0.50
}


class RiskAgent:
    name = "Risk Analysis Agent"
    icon = "⚠️"

    def __init__(self):
        self.predictor = RiskPredictor()

    def run(self, state: dict) -> dict:
        logger.info("RiskAgent running...")

        financials = state.get("financials", DEFAULT_FINANCIALS)

        try:
            result = self.predictor.predict(financials)

            state["risk_result"] = result
            state["agent_logs"].append(
                f"{self.icon} {self.name}: Risk assessed as "
                f"{result['risk_label']} (confidence: {result['confidence']}%)"
            )

        except Exception as e:
            logger.error(f"RiskAgent error: {e}")
            state["risk_result"] = {"risk_label": "Unknown", "risk_color": "#gray", "risk_icon": "❓"}
            state["agent_logs"].append(f"{self.icon} {self.name}: Error — {e}")

        return state