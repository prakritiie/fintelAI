import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from dashboard.ui import apply_theme, page_header

st.set_page_config(page_title="AI Chat", page_icon="💬", layout="wide")

apply_theme()
page_header("AI Chat", "Ask questions about your indexed financial documents. Answers cite their sources.", eyebrow="Step 2")

# ── Session state ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_chain" not in st.session_state:
    try:
        from rag.chain import FinancialRAGChain
        st.session_state.rag_chain = FinancialRAGChain()
    except Exception as e:
        st.error(f"Failed to load RAG chain: {e}")
        st.stop()

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    n_results = st.slider("Chunks to retrieve", 3, 10, 5)
    show_sources = st.toggle("Show source references", value=True)

    st.markdown("### Indexed documents")
    try:
        from core.vector_store import VectorStore
        vs = VectorStore()
        sources = vs.list_sources()
        count = vs.get_collection_count()
        st.info(f"**{count}** chunks across **{len(sources)}** documents")
        for s in sources:
            st.markdown(f"- `{s}`")
    except Exception as e:
        st.warning(f"Could not load sources: {e}")

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### Try asking")
    examples = [
        "What are the major risk factors?",
        "Summarize the revenue performance",
        "What is the company's debt situation?",
        "What are the future growth plans?",
        "How did operating income change?"
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_question = ex

# ── Check documents indexed ──────────────────────────────────
try:
    from core.vector_store import VectorStore
    vs = VectorStore()
    if vs.get_collection_count() == 0:
        st.warning("No documents indexed yet. Please go to **Upload Documents** first.")
        st.stop()
except Exception:
    pass

# ── Chat history ─────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and show_sources and "sources" in message:
            with st.expander("Sources", expanded=False):
                for src in message["sources"]:
                    st.markdown(f"**{src['source']}** | Relevance: `{src['score']}`")
                    st.caption(src["preview"])
                    st.divider()

# ── Handle example question click ────────────────────────────
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.rag_chain.ask(question, n_results=n_results)
                st.markdown(result["answer"])
                if show_sources:
                    with st.expander("Sources", expanded=False):
                        for src in result["sources"]:
                            st.markdown(f"**{src['source']}** | Relevance: `{src['score']}`")
                            st.caption(src["preview"])
                            st.divider()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
            except Exception as e:
                st.error(f"Error: {e}")
    st.rerun()

# ── Chat input ───────────────────────────────────────────────
if question := st.chat_input("Ask about your financial documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            try:
                result = st.session_state.rag_chain.ask(question, n_results=n_results)
                st.markdown(result["answer"])
                if show_sources:
                    with st.expander("Sources", expanded=False):
                        for src in result["sources"]:
                            st.markdown(f"**{src['source']}** | Relevance: `{src['score']}`")
                            st.caption(src["preview"])
                            st.divider()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
            except Exception as e:
                st.error(f"Error: {e}")