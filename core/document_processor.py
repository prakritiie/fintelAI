import fitz  # PyMuPDF
import pdfplumber
import re
from pathlib import Path
from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Section keywords to detect ───────────────────────────────
SECTION_KEYWORDS = {
    "risk_factors":           ["risk factors", "risks and uncertainties"],
    "revenue":                ["revenue", "net sales", "total revenue"],
    "management_discussion":  ["management", "discussion and analysis", "md&a"],
    "financial_statements":   ["financial statements", "balance sheet", "income statement"],
    "future_outlook":         ["outlook", "forward-looking", "future plans"],
}


class DocumentProcessor:

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    # ── 1. Extract raw text ──────────────────────────────────
    def extract_text(self, pdf_path: str) -> str:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Extracting text from: {path.name}")
        full_text = ""

        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text("text")
                    full_text += f"\n[PAGE {page_num + 1}]\n{text}"
            logger.info(f"Extracted {len(full_text)} characters from {path.name}")
        except Exception as e:
            logger.error(f"PyMuPDF failed: {e}. Trying pdfplumber...")
            full_text = self._extract_with_pdfplumber(pdf_path)

        return full_text

    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                full_text += f"\n[PAGE {page_num + 1}]\n{text}"
        logger.info(f"pdfplumber extracted {len(full_text)} characters")
        return full_text

    # ── 2. Clean text ────────────────────────────────────────
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'(\[PAGE \d+\])', r'\n\1\n', text)
        return text.strip()

    # ── 3. Detect sections ───────────────────────────────────
    def detect_sections(self, text: str) -> dict:
        sections = {key: "" for key in SECTION_KEYWORDS}
        lines = text.split('\n')
        current_section = None

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            for section, keywords in SECTION_KEYWORDS.items():
                if any(kw in line_lower for kw in keywords):
                    current_section = section
                    break
            if current_section:
                sections[current_section] += line + "\n"

        detected = [k for k, v in sections.items() if v.strip()]
        logger.info(f"Detected sections: {detected}")
        return sections

    # ── 4. Chunk text ────────────────────────────────────────
    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> list[dict]:
        chunks = self.splitter.split_text(text)
        logger.info(f"Split into {len(chunks)} chunks")

        result = []
        for i, chunk in enumerate(chunks):
            entry = {
                "chunk_id": i,
                "text": chunk,
                "metadata": metadata or {}
            }
            result.append(entry)

        return result

    # ── 5. Full pipeline ─────────────────────────────────────
    def process(self, pdf_path: str, doc_name: Optional[str] = None) -> dict:
        name = doc_name or Path(pdf_path).stem

        raw_text = self.extract_text(pdf_path)
        clean = self.clean_text(raw_text)
        sections = self.detect_sections(clean)

        metadata = {
            "source": name,
            "file_path": pdf_path,
            "total_chars": len(clean)
        }

        chunks = self.chunk_text(clean, metadata=metadata)

        return {
            "doc_name": name,
            "full_text": clean,
            "sections": sections,
            "chunks": chunks,
            "metadata": metadata
        }