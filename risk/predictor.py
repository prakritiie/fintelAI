import pickle
import numpy as np
from pathlib import Path
from utils.logger import get_logger

logger = get_logger(__name__)

FEATURES = [
    "roa", "roe", "current_ratio", "debt_to_equity",
    "profit_margin", "revenue_growth", "interest_coverage", "cash_ratio"
]

RISK_LABELS = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
RISK_COLORS = {0: "#22c55e",  1: "#f59e0b",     2: "#ef4444"}
RISK_ICONS  = {0: "🟢",       1: "🟡",           2: "🔴"}

MODEL_DIR = Path(__file__).resolve().parent / "saved_models"


class RiskPredictor:

    def __init__(self):
        model_path  = MODEL_DIR / "risk_model.pkl"
        scaler_path = MODEL_DIR / "scaler.pkl"

        if not model_path.exists():
            logger.warning("Model not found — training now...")
            from risk.train_model import train
            train()

        with open(model_path,  "rb") as f:
            self.model  = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        logger.info("Risk predictor loaded successfully")

    def predict(self, financials: dict) -> dict:
        values = [[financials[f] for f in FEATURES]]
        scaled = self.scaler.transform(values)

        pred_class = int(self.model.predict(scaled)[0])
        pred_proba = self.model.predict_proba(scaled)[0]

        probabilities = {
            RISK_LABELS[i]: round(float(pred_proba[i]) * 100, 1)
            for i in range(3)
        }

        return {
            "risk_class":     pred_class,
            "risk_label":     RISK_LABELS[pred_class],
            "risk_color":     RISK_COLORS[pred_class],
            "risk_icon":      RISK_ICONS[pred_class],
            "probabilities":  probabilities,
            "confidence":     round(float(pred_proba[pred_class]) * 100, 1),
            "input_features": financials
        }

    def predict_sample_companies(self) -> list[dict]:
        samples = [
            {
                "name": "Apple Inc",
                "roa": 0.19, "roe": 0.35, "current_ratio": 1.8,
                "debt_to_equity": 1.2, "profit_margin": 0.25,
                "revenue_growth": 0.08, "interest_coverage": 18.0,
                "cash_ratio": 0.9
            },
            {
                "name": "Tesla Inc",
                "roa": 0.04, "roe": 0.06, "current_ratio": 1.3,
                "debt_to_equity": 1.8, "profit_margin": 0.08,
                "revenue_growth": 0.19, "interest_coverage": 4.5,
                "cash_ratio": 0.5
            },
            {
                "name": "Risky Corp",
                "roa": -0.08, "roe": -0.15, "current_ratio": 0.6,
                "debt_to_equity": 4.2, "profit_margin": -0.12,
                "revenue_growth": -0.15, "interest_coverage": 0.8,
                "cash_ratio": 0.1
            },
        ]
        results = []
        for s in samples:
            name = s.pop("name")
            result = self.predict(s)
            result["company_name"] = name
            results.append(result)
        return results