"""Chunking strategies for splitting document text before summarization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import tiktoken

from .extractor import Document


class Strategy(str, Enum):
    FIXED = "fixed"        # Split by token count
    PARAGRAPH = "paragraph"  # Split on blank lines, merge up to chunk_size
    PAGE = "page"          # One chunk per page (or N pages)


@dataclass
class Chunk:
    index: int
    text: str
    start_page: int
    end_page: int

    def __str__(self) -> str:
        page_info = (
            f"p{self.start_page}"
            if self.start_page == self.end_page
            else f"p{self.start_page}-{self.end_page}"
        )
        return f"[Chunk {self.index + 1} | {page_info}] {self.text[:80]}..."


def _tokenizer() -> tiktoken.Encoding:
    # cl100k_base is compatible with all current Claude models
    return tiktoken.get_encoding("cl100k_base")


def _token_count(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text))


def _split_fixed(text: str, chunk_size: int, enc: tiktoken.Encoding) -> Iterator[str]:
    """Split text into chunks of at most chunk_size tokens."""
    tokens = enc.encode(text)
    for i in range(0, len(tokens), chunk_size):
        yield enc.decode(tokens[i : i + chunk_size])


def _split_paragraph(text: str, chunk_size: int, enc: tiktoken.Encoding) -> Iterator[str]:
    """Split on paragraph boundaries, merging paragraphs until chunk_size tokens."""
    paragraphs = re.split(r"\n{2,}", text)
    current_tokens: list[int] = []
    current_count = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = enc.encode(para + "\n\n")
        para_count = len(para_tokens)

        if current_count + para_count > chunk_size and current_tokens:
            yield enc.decode(current_tokens)
            current_tokens = para_tokens
            current_count = para_count
        else:
            current_tokens.extend(para_tokens)
            current_count += para_count

    if current_tokens:
        yield enc.decode(current_tokens)


def chunk_document(
    doc: Document,
    strategy: Strategy,
    chunk_size: int,
    pages_per_chunk: int = 1,
) -> list[Chunk]:
    """
    Split a Document into Chunks according to the chosen strategy.

    Args:
        doc: Extracted document.
        strategy: Chunking strategy to use.
        chunk_size: Target token count per chunk (for FIXED and PARAGRAPH).
        pages_per_chunk: Number of pages per chunk (for PAGE strategy).
    """
    enc = _tokenizer()
    chunks: list[Chunk] = []

    if strategy == Strategy.PAGE:
        # Group pages_per_chunk pages together
        i = 0
        chunk_idx = 0
        pages = [p for p in doc.pages if p.text.strip()]
        while i < len(pages):
            group = pages[i : i + pages_per_chunk]
            text = "\n\n".join(p.text for p in group)
            chunks.append(
                Chunk(
                    index=chunk_idx,
                    text=text,
                    start_page=group[0].number,
                    end_page=group[-1].number,
                )
            )
            chunk_idx += 1
            i += pages_per_chunk

    else:
        # For FIXED and PARAGRAPH we work on the full text but track page ranges
        # Build a page-boundary map: character offset -> page number
        full_text = ""
        page_boundaries: list[tuple[int, int]] = []  # (start_offset, page_number)
        for page in doc.pages:
            if page.text.strip():
                page_boundaries.append((len(full_text), page.number))
                full_text += page.text + "\n\n"

        def page_for_offset(offset: int) -> int:
            result = 1
            for start, pnum in page_boundaries:
                if offset >= start:
                    result = pnum
                else:
                    break
            return result

        splitter = _split_fixed if strategy == Strategy.FIXED else _split_paragraph
        char_offset = 0
        for idx, text in enumerate(splitter(full_text, chunk_size, enc)):
            start_page = page_for_offset(char_offset)
            end_page = page_for_offset(char_offset + len(text))
            chunks.append(
                Chunk(index=idx, text=text, start_page=start_page, end_page=end_page)
            )
            char_offset += len(text)

    return chunks
