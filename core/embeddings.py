from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info("Embedding model loaded successfully")

    def embed_text(self, text: str) -> list[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        texts = [c["text"] for c in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()

        logger.info("Embeddings generated successfully")
        return chunks

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(query)