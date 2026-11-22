import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
from dashboard.ui import apply_theme, page_header
import tempfile
import os
from pathlib import Path

st.set_page_config(page_title="Upload Documents", page_icon="📁", layout="wide")

apply_theme()
page_header("Upload Documents", "Index PDF annual reports, 10-K filings or earnings reports so you can query them with AI.", eyebrow="Step 1")

# ── Session state ────────────────────────────────────────────
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []

# ── Sidebar info ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Supported documents")
    st.markdown("""
    - 10-K Annual Reports
    - Earnings Reports
    - SEC Filings
    - Financial Statements
    """)
    st.markdown("### Already indexed")
    try:
        from core.vector_store import VectorStore
        vs = VectorStore()
        sources = vs.list_sources()
        if sources:
            for s in sources:
                st.markdown(f"- `{s}`")
        else:
            st.info("No documents indexed yet")
    except Exception as e:
        st.error(f"Vector store error: {e}")

# ── Upload Section ───────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload multiple PDFs at once"
)

if uploaded_files:
    st.markdown(f"**{len(uploaded_files)} file(s) selected**")

    for uploaded_file in uploaded_files:
        file_size = round(uploaded_file.size / 1024, 1)
        st.markdown(f"- `{uploaded_file.name}` ({file_size} KB)")

    st.markdown("---")

    col1, col2 = st.columns([1, 3])
    with col1:
        process_btn = st.button("Index documents", type="primary", use_container_width=True)
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("This will extract text, generate embeddings, and store in ChromaDB")

    if process_btn:
        from rag.chain import FinancialRAGChain
        chain = FinancialRAGChain()

        for uploaded_file in uploaded_files:
            doc_name = Path(uploaded_file.name).stem

            with st.expander(f"Processing: {uploaded_file.name}", expanded=True):
                progress = st.progress(0, text="Starting...")

                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    progress.progress(20, text="Extracting text...")
                    result = chain.index_document(tmp_path, doc_name)

                    progress.progress(100, text="Done!")

                    st.success(f"Successfully indexed **{doc_name}**")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Chunks Indexed", result["total_chunks"])
                    m2.metric("Sections Found", len(result["sections_detected"]))
                    m3.metric("Total Characters", f"{result['metadata']['total_chars']:,}")

                    st.markdown("**AI-generated summary**")
                    st.markdown(result["summary"])

                    st.markdown("**Detected sections**")
                    st.write(result["sections_detected"])

                    st.session_state.indexed_docs.append(doc_name)

                except Exception as e:
                    st.error(f"Failed to process {uploaded_file.name}: {e}")
                finally:
                    os