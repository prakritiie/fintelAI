from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
from config.settings import FINBERT_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

LABELS = ["positive", "negative", "neutral"]


class FinBERTAnalyzer:

    def __init__(self):
        logger.info(f"Loading FinBERT model: {FINBERT_MODEL}")
        self.tokenizer = BertTokenizer.from_pretrained(FINBERT_MODEL)
        self.model = BertForSequenceClassification.from_pretrained(FINBERT_MODEL)
        self.model.eval()
        logger.info("FinBERT loaded successfully")

    def analyze(self, text: str) -> dict:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = F.softmax(outputs.logits, dim=-1).squeeze()
        scores = {LABELS[i]: round(probs[i].item(), 4) for i in range(3)}
        predicted = max(scores, key=scores.get)

        return {
            "text":      text[:200] + "..." if len(text) > 200 else text,
            "sentiment": predicted,
            "scores":    scores,
            "confidence": round(scores[predicted], 4)
        }

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        logger.info(f"Analyzing sentiment for {len(texts)} texts")
        return [self.analyze(t) for t in texts]

    def analyze_document_chunks(self, chunks: list[str], sample_size: int = 20) -> dict:
        sampled = chunks[:sample_size]
        results = self.analyze_batch(sampled)

        counts = {"positive": 0, "negative": 0, "neutral": 0}
        total_conf = 0.0

        for r in results:
            counts[r["sentiment"]] += 1
            total_conf += r["confidence"]

        total