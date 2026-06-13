import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ── Base Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# ── LLM Settings ────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Embedding Settings ───────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Vector DB Settings ───────────────────────────────────────
CHROMA_DIR = str(BASE_DIR / "data" / "chroma_db")
CHROMA_COLLECTION = "financial_docs"

# ── Sentiment Settings ───────────────────────────────────────
FINBERT_MODEL = "ProsusAI/finbert"

# ── News API ─────────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ── Risk Model ───────────────────────────────────────────────
RISK_MODEL_PATH = str(BASE_DIR / "risk" / "saved_models" / "risk_model.pkl")
RISK_SCALER_PATH = str(BASE_DIR / "risk" / "saved_models" / "scaler.pkl")

# ── App Settings ─────────────────────────────────────────────
APP_TITLE = "FinIntel AI"
APP_ICON = "📊"
DEBUG = os.getenv("DEBUG", "false").lower() == "true"