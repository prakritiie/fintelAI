import shap
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

FEATURES = [
    "roa", "roe", "current_ratio", "debt_to_equity",
    "profit_margin", "revenue_growth", "interest_coverage", "cash_ratio"
]

FEATURE_LABELS = {
    "roa":               "Return on Assets",
    "roe":               "Return on Equity",
    "current_ratio":     "Current Ratio",
    "debt_to_equity":    "Debt to Equity",
    "profit_margin":     "Profit Margin",
    "revenue_growth":    "Revenue Growth",
    "interest_coverage": "Interest Coverage",
    "cash_ratio":        "Cash Ratio"
}


class RiskExplainer:

    def __init__(self, model, scaler):
        self.model     = model
        self.scaler    = scaler
        self.explainer = shap.TreeExplainer(model)

    def explain(self, financials: dict) -> dict:
        values = pd.DataFrame([financials], columns=FEATURES)
        scaled = self.scaler.transform(values)

        shap_values = self.explainer.shap_values(scaled)

        # XGBoost multiclass returns shape (n_samples, n_features, n_classes)
        # or a list of arrays — handle both cases
        pred_class = int(self.model.predict(scaled)[0])

        if isinstance(shap_values, list):
            # List of arrays: one per class
            sv = np.array(shap_values[pred_class]).flatten()
        else:
            # 3D array: (samples, features, classes)
            sv_array = np.array(shap_values)
            if sv_array.ndim == 3:
                sv = sv_array[0, :, pred_class]
            elif sv_array.ndim == 2:
                sv = sv_array[0]
            else:
                sv = sv_array.flatten()

        explanation = []
        for i, feature in enumerate(FEATURES):
            explanation.append({
                "feature":    feature,
                "label":      FEATURE_LABELS[feature],
                "value":      round(float(financials[feature]), 4),
                "shap_value": round(float(sv[i]), 4),
                "impact":     "increases risk" if sv[i] > 0 else "decreases risk"
            })

        explanation.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "predicted_class": pred_class,
            "top_factors":     explanation[:5],
            "all_factors":     explanation
        }