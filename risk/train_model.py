import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import xgboost as xgb
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path(__file__).resolve().parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)


def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    np.random.seed(42)

    # Low risk companies
    low = pd.DataFrame({
        "roa":                np.random.uniform(0.05, 0.25,  n_samples // 3),
        "roe":                np.random.uniform(0.10, 0.40,  n_samples // 3),
        "current_ratio":      np.random.uniform(1.5,  3.5,   n_samples // 3),
        "debt_to_equity":     np.random.uniform(0.1,  0.8,   n_samples // 3),
        "profit_margin":      np.random.uniform(0.10, 0.35,  n_samples // 3),
        "revenue_growth":     np.random.uniform(0.05, 0.30,  n_samples // 3),
        "interest_coverage":  np.random.uniform(5.0,  20.0,  n_samples // 3),
        "cash_ratio":         np.random.uniform(0.5,  2.0,   n_samples // 3),
        "risk_label":         0
    })

    # Medium risk companies
    med = pd.DataFrame({
        "roa":                np.random.uniform(-0.02, 0.08,  n_samples // 3),
        "roe":                np.random.uniform(-0.05, 0.12,  n_samples // 3),
        "current_ratio":      np.random.uniform(0.8,   1.8,   n_samples // 3),
        "debt_to_equity":     np.random.uniform(0.8,   2.0,   n_samples // 3),
        "profit_margin":      np.random.uniform(-0.02, 0.12,  n_samples // 3),
        "revenue_growth":     np.random.uniform(-0.05, 0.08,  n_samples // 3),
        "interest_coverage":  np.random.uniform(1.5,   5.0,   n_samples // 3),
        "cash_ratio":         np.random.uniform(0.2,   0.8,   n_samples // 3),
        "risk_label":         1
    })

    # High risk companies
    high = pd.DataFrame({
        "roa":                np.random.uniform(-0.20, -0.01, n_samples // 3),
        "roe":                np.random.uniform(-0.40, -0.05, n_samples // 3),
        "current_ratio":      np.random.uniform(0.2,   0.9,   n_samples // 3),
        "debt_to_equity":     np.random.uniform(2.0,   6.0,   n_samples // 3),
        "profit_margin":      np.random.uniform(-0.30, -0.02, n_samples // 3),
        "revenue_growth":     np.random.uniform(-0.30, -0.05, n_samples // 3),
        "interest_coverage":  np.random.uniform(0.1,   1.5,   n_samples // 3),
        "cash_ratio":         np.random.uniform(0.0,   0.3,   n_samples // 3),
        "risk_label":         2
    })

    df = pd.concat([low, med, high], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


FEATURES = [
    "roa", "roe", "current_ratio", "debt_to_equity",
    "profit_margin", "revenue_growth", "interest_coverage", "cash_ratio"
]


def train():
    logger.info("Generating training data...")
    df = generate_synthetic_data(1500)

    X = df[FEATURES]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Train XGBoost
    logger.info("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"])
    logger.info(f"\nClassification Report:\n{report}")
    print(report)

    # Save
    with open(MODEL_DIR / "risk_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    logger.info("Model and scaler saved to risk/saved_models/")
    return model, scaler


if __name__ == "__main__":
    train()