"""Chunk store: ingest PDFs, persist chunks as JSON, build BM25 index."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pdfplumber
import tiktoken

from .models import Chunk

# Default store location (sibling to where the tool is invoked)
DEFAULT_STORE = Path(".doc_qa_store.json")

# Token limit per chunk
DEFAULT_CHUNK_SIZE = 400


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

def _extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return list of (page_number, text) tuples."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
    return pages


# ---------------------------------------------------------------------------
# Chunking (paragraph-aware, token-bounded)
# ---------------------------------------------------------------------------

def _tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int,
) -> list[tuple[int, int, str]]:
    """
    Split pages into (start_page, end_page, text) chunks.
    Merges paragraphs up to chunk_size tokens, never splits mid-paragraph.
    """
    enc = _tokenizer()
    chunks: list[tuple[int, int, str]] = []
    current_parts: list[str] = []
    current_tokens = 0
    current_start = pages[0][0] if pages else 1
    current_end = current_start

    for page_num, page_text in pages:
        paragraphs = re.split(r"\n{2,}", page_text)
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            count = len(enc.encode(para))
            if current_tokens + count > chunk_size and current_parts:
                chunks.append((current_start, current_end, "\n\n".join(current_parts)))
                current_parts = [para]
                current_tokens = count
                current_start = page_num
                current_end = page_num
            else:
                current_parts.append(para)
                current_tokens += count
                current_end = page_num

    if current_parts:
        chunks.append((current_start, current_end, "\n\n".join(current_parts)))

    return chunks


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode()
    name = re.sub(r"[^\w\s-]", "", name).strip().lower()
    return re.sub(r"[\s_-]+", "_", name)


# ---------------------------------------------------------------------------
# Store (JSON-backed)
# ---------------------------------------------------------------------------

class ChunkStore:
    """
    Persistent store of document chunks.

    Backed by a single JSON file. Supports adding documents and iterating chunks.
    """

    def __init__(self, store_path: Path = DEFAULT_STORE) -> None:
        self.store_path = store_path
        self._chunks: dict[str, dict] = {}  # id -> serialized Chunk
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            with open(self.store_path, encoding="utf-8") as f:
                self._chunks = json.load(f)

    def _save(self) -> None:
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self._chunks, f, ensure_ascii=False, indent=2)

    def add_document(
        self,
        pdf_path: Path,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        overwrite: bool = False,
    ) -> list[Chunk]:
        """Extract, chunk, and store a PDF. Returns the new Chunk objects."""
        slug = _slugify(pdf_path.stem)
        existing = [c for c in self._chunks.values() if c["doc_path"] == str(pdf_path.resolve())]
        if existing and not overwrite:
            return [self._deserialize(c) for c in existing]

        # Remove old chunks for this doc if re-indexing
        if overwrite:
            self._chunks = {
                k: v for k, v in self._chunks.items()
                if v["doc_path"] != str(pdf_path.resolve())
            }

        pages = _extract_pages(pdf_path)
        raw_chunks = _chunk_pages(pages, chunk_size)

        new_chunks: list[Chunk] = []
        for idx, (start, end, text) in enumerate(raw_chunks):
            chunk_id = f"{slug}__p{start}-{end}__{idx}"
            chunk = Chunk(
                id=chunk_id,
                doc_name=pdf_path.name,
                doc_path=str(pdf_path.resolve()),
                start_page=start,
                end_page=end,
                chunk_index=idx,
                text=text,
            )
            self._chunks[chunk_id] = self._serialize(chunk)
            new_chunks.append(chunk)

        self._save()
        return new_chunks

    def all_chunks(self) -> list[Chunk]:
        return [self._deserialize(v) for v in self._chunks.values()]

    def indexed_docs(self) -> list[str]:
        seen: dict[str, str] = {}
        for v in self._chunks.values():
            seen[v["doc_path"]] = v["doc_name"]
        return [f"{name}  ({path})" for path, name in seen.items()]

    def chunk_count(self) -> int:
        return len(self._chunks)

    @staticmethod
    def _serialize(c: Chunk) -> dict:
        return {
            "id": c.id,
            "doc_name": c.doc_name,
            "doc_path": c.doc_path,
            "start_page": c.start_page,
            "end_page": c.end_page,
            "chunk_index": c.chunk_index,
            "text": c.text,
        }

    @staticmethod
    def _deserialize(d: dict) -> Chunk:
        return Chunk(**d)
