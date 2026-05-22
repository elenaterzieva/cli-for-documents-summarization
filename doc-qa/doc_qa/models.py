"""Core data models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A text chunk from a document."""
    id: str                  # "<doc_slug>__p<start>-<end>__<idx>"
    doc_name: str            # Original filename
    doc_path: str            # Absolute path at index time
    start_page: int
    end_page: int
    chunk_index: int
    text: str

    def citation_label(self) -> str:
        page = (
            f"p{self.start_page}"
            if self.start_page == self.end_page
            else f"p{self.start_page}–{self.end_page}"
        )
        return f"{self.doc_name} [{page}]"


@dataclass
class Citation:
    """A single source reference used in an answer."""
    chunk_id: str
    doc_name: str
    start_page: int
    end_page: int
    relevance_score: float
    excerpt: str             # First 200 chars of the chunk

    def label(self) -> str:
        page = (
            f"p{self.start_page}"
            if self.start_page == self.end_page
            else f"p{self.start_page}–{self.end_page}"
        )
        return f"{self.doc_name} [{page}]"


@dataclass
class SubAnswer:
    """Answer to one sub-question in a multi-hop chain."""
    sub_question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)


@dataclass
class Answer:
    """Final answer to the user's question."""
    question: str
    final_answer: str
    sub_answers: list[SubAnswer] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    is_multi_hop: bool = False
    total_input_tokens: int = 0
    total_output_tokens: int = 0
