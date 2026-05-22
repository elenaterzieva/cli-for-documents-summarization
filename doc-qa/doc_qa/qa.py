"""Multi-hop Q&A engine with citation tracking."""

from __future__ import annotations

import json
from typing import Callable

import anthropic

from .models import Answer, Citation, Chunk, SubAnswer
from .retriever import Retriever

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM = """\
You are an expert at breaking down complex questions into simpler sub-questions.
Given a question, determine if it requires information from multiple sources or sections (multi-hop).
If so, output a JSON list of sub-questions that together would answer the original question.
If the question can be answered directly, output a JSON list with just the original question.

Output ONLY valid JSON — a list of strings. No explanation, no markdown fences."""

DECOMPOSE_USER = """Question: {question}

Output a JSON list of sub-questions (1 to 4 items)."""

SUB_ANSWER_SYSTEM = """\
You are a precise question-answering assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say so clearly.
Be concise and factual."""

SUB_ANSWER_USER = """\
Context (from documents):
{context}

Question: {question}

Answer based solely on the context above."""

SYNTHESIZE_SYSTEM = """\
You are an expert at synthesizing information from multiple sources into a clear, coherent answer.
You will receive a complex question and a series of sub-answers derived from different document sections.
Produce a final, unified answer that directly addresses the original question.
Be accurate, concise, and cite which sub-answers informed each part of your response."""

SYNTHESIZE_USER = """\
Original question: {question}

Sub-answers gathered from the documents:
{sub_answers}

Write a final, comprehensive answer to the original question."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_context(chunks: list[tuple[Chunk, float]]) -> str:
    parts = []
    for chunk, _ in chunks:
        parts.append(f"[{chunk.citation_label()}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def _call(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
) -> tuple[str, int, int]:
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text, resp.usage.input_tokens, resp.usage.output_tokens


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class QAEngine:
    """
    Multi-hop Q&A engine.

    Pipeline:
      1. Decompose the question into sub-questions (via Claude).
      2. Retrieve relevant chunks per sub-question (BM25).
      3. Answer each sub-question with its retrieved context.
      4. Synthesize a final answer from all sub-answers.
      5. Deduplicate and return all citations.
    """

    def __init__(
        self,
        retriever: Retriever,
        model: str = "claude-sonnet-4-6",
        top_k: int = 4,
        max_sub_tokens: int = 512,
        max_final_tokens: int = 1024,
    ) -> None:
        self.retriever = retriever
        self.model = model
        self.top_k = top_k
        self.max_sub_tokens = max_sub_tokens
        self.max_final_tokens = max_final_tokens
        self._client = anthropic.Anthropic()

    def ask(
        self,
        question: str,
        doc_filter: list[str] | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> Answer:
        in_tok = out_tok = 0

        def log(msg: str) -> None:
            if progress:
                progress(msg)

        # --- Step 1: Decompose ---
        log("Decomposing question into sub-questions...")
        raw, i, o = _call(
            self._client,
            self.model,
            DECOMPOSE_SYSTEM,
            DECOMPOSE_USER.format(question=question),
            max_tokens=256,
        )
        in_tok += i
        out_tok += o

        try:
            sub_questions: list[str] = json.loads(raw)
            if not isinstance(sub_questions, list) or not sub_questions:
                sub_questions = [question]
        except (json.JSONDecodeError, ValueError):
            sub_questions = [question]

        is_multi_hop = len(sub_questions) > 1
        log(f"{'Multi-hop' if is_multi_hop else 'Single-hop'}: {len(sub_questions)} sub-question(s)")

        # --- Steps 2 & 3: Retrieve + answer each sub-question ---
        sub_answers: list[SubAnswer] = []
        all_citations: dict[str, Citation] = {}  # chunk_id -> Citation (deduped)

        for idx, sq in enumerate(sub_questions):
            log(f"Answering sub-question {idx + 1}/{len(sub_questions)}: {sq[:60]}...")
            results = self.retriever.retrieve(sq, top_k=self.top_k, doc_filter=doc_filter)

            if not results:
                sub_answers.append(SubAnswer(sub_question=sq, answer="No relevant information found."))
                continue

            context = _format_context(results)
            raw_ans, i, o = _call(
                self._client,
                self.model,
                SUB_ANSWER_SYSTEM,
                SUB_ANSWER_USER.format(context=context, question=sq),
                max_tokens=self.max_sub_tokens,
            )
            in_tok += i
            out_tok += o

            citations = self.retriever.to_citations(results)
            for c in citations:
                all_citations[c.chunk_id] = c

            sub_answers.append(SubAnswer(sub_question=sq, answer=raw_ans, citations=citations))

        # --- Step 4: Synthesize final answer ---
        log("Synthesizing final answer...")
        if len(sub_answers) == 1:
            final = sub_answers[0].answer
        else:
            sub_block = "\n\n".join(
                f"Sub-question {i + 1}: {sa.sub_question}\nAnswer: {sa.answer}"
                for i, sa in enumerate(sub_answers)
            )
            final, i, o = _call(
                self._client,
                self.model,
                SYNTHESIZE_SYSTEM,
                SYNTHESIZE_USER.format(question=question, sub_answers=sub_block),
                max_tokens=self.max_final_tokens,
            )
            in_tok += i
            out_tok += o

        # Sort citations by relevance score descending
        sorted_citations = sorted(
            all_citations.values(),
            key=lambda c: c.relevance_score,
            reverse=True,
        )

        return Answer(
            question=question,
            final_answer=final,
            sub_answers=sub_answers,
            citations=sorted_citations,
            is_multi_hop=is_multi_hop,
            total_input_tokens=in_tok,
            total_output_tokens=out_tok,
        )
