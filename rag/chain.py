from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from rag.prompts import FINANCIAL_QA_PROMPT, SUMMARY_PROMPT
from rag.retriever import FinancialRetriever
from config.settings import GROQ_API_KEY, GROQ_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)


class FinancialRAGChain:

    def __init__(self):
        self.retriever = FinancialRetriever()
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.1
        )
        self.parser = StrOutputParser()
        logger.info(f"RAG chain initialized with model: {GROQ_MODEL}")

    # ── Q&A ──────────────────────────────────────────────────
    def ask(self, question: str, n_results: int = 5) -> dict:
        logger.info(f"Question: {question}")
        context, chunks = self.retriever.retrieve_with_context(question, n_results)

        chain = FINANCIAL_QA_PROMPT | self.llm | self.parser
        answer = chain.invoke({
            "context": context,
            "question": question
        })

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "source": c["metadata"].get("source", "Unknown"),
                    "score": c.get("score", 0),
                    "preview": c["text"][:200] + "..."
                }
                for c in chunks
            ]
        }

    # ── Summarize ────────────────────────────────────────────
    def summarize(self, doc_name: str, content: str) -> str:
        logger.info(f"Summarizing: {doc_name}")
        chain = SUMMARY_PROMPT | self.llm | self.parser
        summary = chain.invoke({
            "doc_name": doc_name,
            "content": content[:6000]          # stay within token limit
        })
        return summary

    # ── Full pipeline: upload + index ─────────────────────────
    def index_document(self, pdf_path: str, doc_name: str = None) -> dict:
        from core.document_processor import DocumentProcessor
        from core.embeddings import EmbeddingGenerator
        from core.vector_store import VectorStore

        processor = DocumentProcessor()
        embedder = EmbeddingGenerator()
        vector_store = VectorStore()

        logger.info(f"Indexing document: {pdf_path}")

        result = processor.process(pdf_path, doc_name)
        chunks_with_embeddings = embedder.embed_chunks(result["chunks"])
        vector_store.add_chunks(chunks_with_embeddings)

        summary = self.summarize(
            result["doc_name"],
            result["full_text"]
        )

        logger.info(f"Document indexed successfully: {result['doc_name']}")

        return {
            "doc_name": result["doc_name"],
            "total_chunks": len(chunks_with_embeddings),
            "summary": summary,
            "sections_detected": list(result["sections"].keys()),
            "metadata": result["metadata"]
        }