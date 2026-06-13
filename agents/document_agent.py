from rag.retriever import FinancialRetriever
from rag.chain import FinancialRAGChain
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentAgent:
    name = "Document Analysis Agent"
    icon = "📄"

    def __init__(self):
        self.retriever = FinancialRetriever()
        self.chain     = FinancialRAGChain()

    def run(self, state: dict) -> dict:
        logger.info("DocumentAgent running...")
        doc_name = state.get("doc_name", "Unknown Document")

        try:
            vs = self.retriever.vector_store
            count = vs.get_collection_count()

            if count == 0:
                state["document_analysis"] = "No documents indexed yet."
                state["agent_logs"].append(f"{self.icon} {self.name}: No documents found in vector store.")
                return state

            # Pull key sections via targeted queries
            queries = [
                "revenue and financial performance",
                "risk factors and challenges",
                "future outlook and guidance",
                "management discussion and analysis"
            ]

            sections = {}
            for q in queries:
                context, _ = self.retriever.retrieve_with_context(q, n_results=3)
                sections[q] = context[:1000]

            combined = "\n\n".join([f"### {k}\n{v}" for k, v in sections.items()])

            summary = self.chain.summarize(doc_name, combined)

            state["document_analysis"] = summary
            state["sections"]          = sections
            state["agent_logs"].append(
                f"{self.icon} {self.name}: Analyzed {count} chunks, extracted {len(sections)} sections."
            )

        except Exception as e:
            logger.error(f"DocumentAgent error: {e}")
            state["document_analysis"] = f"Error during document analysis: {e}"
            state["agent_logs"].append(f"{self.icon} {self.name}: Error — {e}")

        return state