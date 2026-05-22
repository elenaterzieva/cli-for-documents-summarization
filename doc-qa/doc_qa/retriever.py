"""BM25-based retrieval over indexed chunks."""

from __future__ import annotations

import re
import string

from rank_bm25 import BM25Okapi

from .models import Chunk, Citation


def _tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()


class Retriever:
    """
    Builds a BM25 index over a list of Chunk objects and supports
    top-k retrieval for a query string.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: list[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """
        Return up to top_k (Chunk, score) pairs ranked by BM25 relevance.

        Args:
            query: The search query.
            top_k: Number of results to return.
            doc_filter: If set, only return chunks from these document names.
        """
        if not self._bm25 or not self.chunks:
            return []

        tokens = _tokenize(query)
        scores: list[float] = self._bm25.get_scores(tokens)

        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        if doc_filter:
            ranked = [(c, s) for c, s in ranked if c.doc_name in doc_filter]

        return [(c, float(s)) for c, s in ranked[:top_k]]

    def to_citations(
        self,
        results: list[tuple[Chunk, float]],
    ) -> list[Citation]:
        """Convert retrieval results to Citation objects."""
        citations = []
        for chunk, score in results:
            citations.append(
                Citation(
                    chunk_id=chunk.id,
                    doc_name=chunk.doc_name,
                    start_page=chunk.start_page,
                    end_page=chunk.end_page,
                    relevance_score=score,
                    excerpt=chunk.text[:200].replace("\n", " "),
                )
            )
        return citations
