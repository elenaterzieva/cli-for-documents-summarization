"""Hierarchical summarization using the Anthropic Claude API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import anthropic

from .chunker import Chunk

# Default prompts — users can override via CLI flags
CHUNK_SYSTEM = (
    "You are an expert summarizer. Produce a concise, information-dense summary "
    "of the provided text excerpt, preserving all key facts, figures, and arguments."
)

FINAL_SYSTEM = (
    "You are an expert summarizer. You will receive a series of partial summaries "
    "from sections of a longer document. Synthesize them into a single, coherent, "
    "well-structured final summary that captures the essential content of the whole document."
)

CHUNK_USER_TEMPLATE = "Summarize the following excerpt:\n\n{text}"

FINAL_USER_TEMPLATE = (
    "Below are summaries of {n} sections of the document \"{title}\".\n\n"
    "{summaries}\n\n"
    "Write a unified final summary of the entire document."
)


@dataclass
class SummaryResult:
    model: str
    strategy: str
    chunk_size: int
    num_chunks: int
    chunk_summaries: list[str] = field(default_factory=list)
    final_summary: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0


def _call_claude(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
) -> tuple[str, int, int]:
    """Call Claude and return (text, input_tokens, output_tokens)."""
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = response.content[0].text
    return text, response.usage.input_tokens, response.usage.output_tokens


def summarize(
    chunks: list[Chunk],
    model: str,
    title: str,
    strategy: str,
    chunk_size: int,
    max_chunk_tokens: int = 512,
    max_final_tokens: int = 1024,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> SummaryResult:
    """
    Hierarchically summarize a list of chunks.

    1. Summarize each chunk individually.
    2. Combine chunk summaries into a final summary.

    Args:
        chunks: List of document chunks.
        model: Claude model ID.
        title: Document title (filename) for the final prompt.
        strategy: Chunking strategy name (for the result metadata).
        chunk_size: Chunk size (for the result metadata).
        max_chunk_tokens: Max tokens for each chunk summary response.
        max_final_tokens: Max tokens for the final summary response.
        progress_callback: Optional fn(current, total, message) for progress reporting.
    """
    client = anthropic.Anthropic()
    result = SummaryResult(
        model=model,
        strategy=strategy,
        chunk_size=chunk_size,
        num_chunks=len(chunks),
    )

    # --- Step 1: Summarize each chunk ---
    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(i + 1, len(chunks), f"Summarizing chunk {i + 1}/{len(chunks)}")

        user_msg = CHUNK_USER_TEMPLATE.format(text=chunk.text)
        summary, in_tok, out_tok = _call_claude(
            client, model, CHUNK_SYSTEM, user_msg, max_chunk_tokens
        )
        result.chunk_summaries.append(summary)
        result.total_input_tokens += in_tok
        result.total_output_tokens += out_tok

    # --- Step 2: Final synthesis ---
    if progress_callback:
        progress_callback(len(chunks), len(chunks), "Synthesizing final summary...")

    if len(result.chunk_summaries) == 1:
        # Only one chunk — the chunk summary IS the final summary
        result.final_summary = result.chunk_summaries[0]
    else:
        numbered = "\n\n".join(
            f"[Section {i + 1}]\n{s}" for i, s in enumerate(result.chunk_summaries)
        )
        user_msg = FINAL_USER_TEMPLATE.format(
            n=len(result.chunk_summaries),
            title=title,
            summaries=numbered,
        )
        final, in_tok, out_tok = _call_claude(
            client, model, FINAL_SYSTEM, user_msg, max_final_tokens
        )
        result.final_summary = final
        result.total_input_tokens += in_tok
        result.total_output_tokens += out_tok

    return result
