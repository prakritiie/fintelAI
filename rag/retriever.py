import math
import re
from collections import Counter
from core.embeddings import EmbeddingGenerator
from core.vector_store import VectorStore
from utils.logger import get_logger

logger = get_logger(__name__)

# Everyday phrasing -> wording commonly used in SEC filings.
# Used only to widen the keyword side of the hybrid search.
_SYNONYMS = {
    "operating income":   ["income from operations", "operating income"],
    "operating profit":   ["income from operations"],
    "net income":         ["net income", "net income attributable"],
    "profit":             ["net income", "gross profit"],
    "sales":              ["revenues", "automotive sales"],
    "revenue":            ["total revenues", "revenues"],
    "debt":               ["indebtedness", "debt and finance leases", "notes"],
    "cash":               ["cash and cash equivalents", "cash flows"],
    "capex":              ["capital expenditures"],
    "eps":                ["net income per share", "per share"],
    "employees":          ["headcount", "employee"],
}
# Exact wording matches (numbers, statement line items) are what embeddings miss,
# so the keyword ranking counts double in the fusion.
_KW_WEIGHT = 2.0
_RESERVED_KEYWORD_SLOTS = 2
_STOP = {"what", "was", "were", "is", "are", "the", "a", "an", "of", "in", "for", "to", "and",
         "how", "did", "do", "does", "company", "companys", "s", "about", "me", "tell", "summarize"}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9$%.,]+", text.lower()) if t not in _STOP and len(t) > 1]


class FinancialRetriever:

    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self._corpus_count = -1
        self._corpus: list[dict] = []
        self._df: Counter = Counter()

    # ── keyword side ─────────────────────────────────────────
    def _load_corpus(self) -> list[dict]:
        count = self.vector_store.get_collection_count()
        if count != self._corpus_count:
            data = self.vector_store.collection.get(include=["documents", "metadatas"])
            self._corpus = [
                {"text": t, "metadata": m, "tokens": set(_tokens(t)), "lower": t.lower()}
                for t, m in zip(data["documents"], data["metadatas"])
            ]
            self._df = Counter(t for c in self._corpus for t in c["tokens"])
            self._corpus_count = count
        return self._corpus

    def _expand(self, query: str) -> tuple[set[str], list[str]]:
        q = query.lower()
        phrases = [p for k, v in _SYNONYMS.items() if k in q for p in v]
        return set(_tokens(q)), phrases

    def _keyword_rank(self, query: str, k: int) -> list[dict]:
        terms, phrases = self._expand(query)
        if not terms:
            return []
        corpus = self._load_corpus()
        n = len(corpus)
        idf = {t: math.log(1 + n / (1 + self._df.get(t, 0))) for t in terms}
        total = sum(idf.values()) or 1.0
        scored = []
        for c in corpus:
            score = sum(idf[t] for t in terms & c["tokens"]) / total
            score += 1.5 * sum(p in c["lower"] for p in phrases)  # phrase hits matter most
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]

    # ── public API ───────────────────────────────────────────
    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        logger.info(f"Retrieving chunks for: '{query}'")
        pool = max(n_results * 4, 20)
        query_embedding = self.embedder.embed_query(query)
        vec = self.vector_store.query(query_embedding, n_results=pool)

        try:
            kw = self._keyword_rank(query, pool)
        except Exception as e:  # never let the keyword side break plain vector search
            logger.warning(f"Keyword search skipped: {e}")
            kw = []

        # Reciprocal rank fusion of the two ranked lists
        fused: dict[str, dict] = {}
        for rank, c in enumerate(vec):
            key = c["text"]
            fused[key] = {"chunk": c, "rrf": 1 / (60 + rank)}
        for rank, c in enumerate(kw):
            key = c["text"]
            if key in fused:
                fused[key]["rrf"] += _KW_WEIGHT / (60 + rank)
            else:
                fused[key] = {
                    "chunk": {"text": c["text"], "metadata": c["metadata"], "score": "keyword match"},
                    "rrf": _KW_WEIGHT / (60 + rank),
                }

        ranked = sorted(fused.values(), key=lambda x: x["rrf"], reverse=True)

        # Guarantee the very best exact-wording matches a slot: a chunk that is only
        # strong on keywords (e.g. an income-statement table) would otherwise lose to
        # chunks that merely appear in both lists.
        reserve = min(_RESERVED_KEYWORD_SLOTS, n_results // 2)
        final = [fused[c["text"]]["chunk"] for c in kw[:reserve]]
        seen = {c["text"] for c in final}
        for item in ranked:
            if len(final) >= n_results:
                break
            if item["chunk"]["text"] not in seen:
                final.append(item["chunk"])
                seen.add(item["chunk"]["text"])
        return final

    def format_context(self, chunks: list[dict]) -> str:
        context_parts = []
        for i, chunk in enumerate(chunks):
            source = chunk["metadata"].get("source", "Unknown")
            score = chunk.get("score", 0)
            context_parts.append(
                f"[Source: {source} | Relevance: {score}]\n{chunk['text']}"
            )
        return "\n\n---\n\n".join(context_parts)

    def retrieve_with_context(self, query: str, n_results: int = 5) -> tuple[str, list[dict]]:
        chunks = self.retrieve(query, n_results)
        context = self.format_context(chunks)
        return context, chunks
