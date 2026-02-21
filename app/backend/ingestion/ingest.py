"""
Data ingestion module for processing documentation into searchable indices.

Implements section-aware chunking by detecting heading boundaries in PDFs 
to maintain semantic coherence. Generates both a FAISS IndexFlatIP for 
dense vector retrieval and a BM25 index for sparse keyword matching.
"""
import os
import json
import pickle
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    print("WARNING: pypdf not installed. Fallback for EOF PDFs disabled.")


# Handle relative paths for reproducibility
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

PDF_DIR       = os.path.join(BASE_DIR, "docs")
INDEX_PATH    = os.path.join(os.path.dirname(__file__), "../index.faiss")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "../chunk_metadata.json")
BM25_PATH     = os.path.join(os.path.dirname(__file__), "../bm25_index.pkl")

# Chunk size tuned for these short synthetic PDFs.
# At 2400 chars most small docs (700-1600 chars) were exactly 1 chunk.
# At 900 chars we get 1-2 chunks for small docs and ~10-12 for larger ones
# ensuring retrieval targets specific passages rather than returning near-entire documents.
MAX_CHUNK_CHARS = 900    # ~225 tokens (tight, specific)
OVERLAP_CHARS   = 180    # ~45 tokens overlap (context continuity)



# Section-aware PDF extraction logic
def _is_heading(chars) -> bool:
    """Heuristic: a line is a heading if its average font size > 12"""
    if not chars:
        return False
    avg_size = sum(c.get("size", 0) or 0 for c in chars) / len(chars)
    return avg_size > 12


def _table_to_text(table: list) -> str:
    """
    Convert a pdfplumber table (list of rows, each a list of cells)
    into a pipe-delimited readable text block.
    Empty cells are replaced with '-'
    """
    if not table:
        return ""
    rows = []
    for row in table:
        cells = [str(cell).strip() if cell is not None else "-" for cell in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def _extract_with_pypdf(pdf_path: str) -> str:
    """
    Fallback extractor using pypdf — much more lenient on malformed/EOF PDFs.
    Used when pdfplumber throws any exception
    """
    if not PYPDF_AVAILABLE:
        return ""
    text = ""
    try:
        reader = PdfReader(pdf_path, strict=False)  #tolerates EOF issues
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text.strip() + "\n"
        print(f"    [pypdf fallback] Extracted {len(text)} chars from {len(reader.pages)} pages.")
    except Exception as e:
        print(f"    [pypdf fallback] Also failed: {e}")
    return text


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Robust PDF text extractor with a two-tier strategy:
      Tier 1 — pdfplumber: rich extraction including table cells (pipe-delimited).
      Tier 2 — pypdf (fallback): lenient parser that handles PDFs pdfplumber
               rejects
    """
    full_text = ""
    pdfplumber_ok = False

    #pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_body   = ""
                page_tables = ""

                try:
                    page_body = page.extract_text() or ""
                    if page_body:
                        page_body = page_body.strip() + "\n"
                except Exception as body_err:
                    print(f"    Page {page_num+1} body error: {body_err}")

                try:
                    tables = page.extract_tables()
                    for tbl in tables:
                        table_text = _table_to_text(tbl)
                        if table_text.strip():
                            page_tables += "\n[TABLE]\n" + table_text + "\n[/TABLE]\n"
                except Exception as tbl_err:
                    print(f"    Page {page_num+1} table error: {tbl_err}")

                full_text += page_body + page_tables
        pdfplumber_ok = True
    except Exception as e:
        print(f"  pdfplumber failed ({e}), trying pypdf fallback...")

    #pypdf fallback
    if not pdfplumber_ok or not full_text.strip():
        fallback_text = _extract_with_pypdf(pdf_path)
        if fallback_text.strip():
            full_text = fallback_text

    if not full_text.strip():
        print(f"  Warning: Both extractors failed for {os.path.basename(pdf_path)} — skipping.")

    return full_text



def extract_sections_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Wraps extract_text_from_pdf into the section list format the rest of
    the pipeline expects: [{section_title, text}, ...].
    We return a single section per document (reliable), and rely on
    chunk_section to do the fine-grained token-cap splitting.
    """
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        return []
    return [{"section_title": "Document", "text": text.strip()}]


def chunk_section(section_text: str, doc_id: str, section_title: str,
                  base_chunk_idx: int) -> List[Dict]:
    """Sliding-window chunk a single section's text."""
    chunks       = []
    start        = 0
    text_len     = len(section_text)
    chunk_idx    = base_chunk_idx

    while start < text_len:
        end = min(start + MAX_CHUNK_CHARS, text_len)

        if end < text_len:
            snap_nl = section_text.rfind("\n", start, end)
            snap_pe = section_text.rfind(". ", start, end)
            if snap_nl != -1 and snap_nl > start + MAX_CHUNK_CHARS // 2:
                end = snap_nl + 1
            elif snap_pe != -1 and snap_pe > start + MAX_CHUNK_CHARS // 2:
                end = snap_pe + 2

        chunk_text = section_text[start:end].strip()
        if len(chunk_text) > 50:
            chunks.append({
                "doc_id":        doc_id,
                "chunk_id":      f"{doc_id}_chunk_{chunk_idx}",
                "section_title": section_title,
                "text":          chunk_text,
            })
            chunk_idx += 1

        start = end - OVERLAP_CHARS
        if start < 0 or end == text_len:
            break

    return chunks


# BM25 Index Management
def _build_and_save_bm25(all_chunks: List[Dict]):
    try:
        from rank_bm25 import BM25Okapi
        import re

        def tokenize(text):
            return re.sub(r'[^a-z0-9\s]', ' ', text.lower()).split()

        tokenized = [tokenize(c["text"]) for c in all_chunks]
        bm25 = BM25Okapi(tokenized)
        with open(BM25_PATH, "wb") as f:
            pickle.dump({"index": bm25, "corpus": [c["text"] for c in all_chunks]}, f)
        print(f"BM25 index saved to {BM25_PATH}")
    except ImportError:
        print("rank-bm25 not installed — skipping BM25 index. Install with: pip install rank-bm25")


# Main ingestion orchestrator
def run_ingestion():
    print("Loading embedding model (all-mpnet-base-v2)...")
    embedder = SentenceTransformer("all-mpnet-base-v2")

    all_chunks: List[Dict] = []

    print(f"Scanning {PDF_DIR} for PDFs...")
    pdf_files = sorted(f for f in os.listdir(PDF_DIR) if f.endswith(".pdf"))

    for pdf_file in pdf_files:
        print(f"  Processing {pdf_file}...")
        chunks_for_doc = []
        sections = extract_sections_from_pdf(os.path.join(PDF_DIR, pdf_file))

        chunk_idx = 0
        for section in sections:
            new_chunks = chunk_section(
                section["text"],
                doc_id        = pdf_file,
                section_title = section["section_title"],
                base_chunk_idx = chunk_idx
            )
            chunks_for_doc.extend(new_chunks)
            chunk_idx += len(new_chunks)

        print(f"    → {len(chunks_for_doc)} chunks from {len(sections)} sections")
        all_chunks.extend(chunks_for_doc)

    print(f"\nTotal chunks: {len(all_chunks)}")

    texts_to_embed = [c["text"] for c in all_chunks]
    print("Generating embeddings...")
    embeddings = embedder.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=True, batch_size=32)
    faiss.normalize_L2(embeddings)

    print("Building FAISS IndexFlatIP...")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved ({index.ntotal} vectors) → {INDEX_PATH}")

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    print(f"Metadata saved → {METADATA_PATH}")

    print("Building BM25 keyword index...")
    _build_and_save_bm25(all_chunks)

    print("\nIngestion complete.")


if __name__ == "__main__":
    run_ingestion()