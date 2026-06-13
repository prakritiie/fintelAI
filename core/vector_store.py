import chromadb
from chromadb.config import Settings
from config.settings import CHROMA_DIR, CHROMA_COLLECTION
from utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:

    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION,
        persist_dir: str = CHROMA_DIR
    ):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"VectorStore ready — collection: '{collection_name}'")

    # ── Add documents ────────────────────────────────────────
    def add_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            logger.warning("No chunks to add")
            return

        ids, documents, embeddings, metadatas = [], [], [], []

        for chunk in chunks:
            chunk_id = f"{chunk['metadata'].get('source', 'doc')}_{chunk['chunk_id']}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            embeddings.append(chunk["embedding"])
            metadatas.append(chunk["metadata"])

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        logger.info(f"Added {len(chunks)} chunks to vector store")

    # ── Query ────────────────────────────────────────────────
    def query(self, query_embedding: list[float], n_results: int = 5) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        output = []
        for i in range(len(results["documents"][0])):
            output.append({
                "text":     results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score":    round(1 - results["distances"][0][i], 4)
            })

        logger.info(f"Retrieved {len(output)} chunks for query")
        return output

    # ── Utils ────────────────────────────────────────────────
    def get_collection_count(self) -> int:
        return self.collection.count()

    def delete_collection(self) -> None:
        self.client.delete_collection(self.collection.name)
        logger.info("Collection deleted")

    def list_sources(self) -> list[str]:
        results = self.collection.get(include=["metadatas"])
        sources = list({m.get("source", "unknown") for m in results["metadatas"]})
        return sources