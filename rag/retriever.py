from core.embeddings import EmbeddingGenerator
from core.vector_store import VectorStore
from utils.logger import get_logger

logger = get_logger(__name__)


class FinancialRetriever:

    def __init__(self):
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()

    def retrieve(self, query: str, n_results: int = 5) -> list[dict]:
        logger.info(f"Retrieving chunks for: '{query}'")
        query_embedding = self.embedder.embed_query(query)
        chunks = self.vector_store.query(query_embedding, n_results=n_results)
        return chunks

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