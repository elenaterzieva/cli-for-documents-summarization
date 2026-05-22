# Document Q&A with Citation Tracking

A CLI tool for asking questions across multiple PDF documents. It handles **multi-hop questions** — questions that require synthesizing information from multiple documents or sections — and tracks exactly which sources were used to produce every answer.

## How it works

1. **Index** — PDFs are extracted page-by-page and split into overlapping paragraph-aware chunks stored in a local JSON index.
2. **Decompose** — When you ask a question, Claude decides whether it's multi-hop and breaks it into sub-questions.
3. **Retrieve** — Each sub-question is answered using BM25 retrieval over the indexed chunks.
4. **Synthesize** — Sub-answers are combined into a single final answer.
5. **Cite** — Every chunk used is tracked and returned as a citation with document name, page range, and relevance score.

## Installation

Requires Python 3.10+.

```bash
cd doc-qa
pip install -e .
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_key_here   # macOS/Linux
set ANTHROPIC_API_KEY=your_key_here      # Windows
```

## Usage

### 1. Index documents

```bash
# Index one or more PDFs
doc-qa index paper1.pdf paper2.pdf report.pdf

# Custom chunk size and store location
doc-qa index *.pdf --chunk-size 600 --store my_index.json

# Re-index a document that changed
doc-qa index paper1.pdf --overwrite
```

### 2. List indexed documents

```bash
doc-qa list
```

### 3. Ask questions

```bash
# Simple question
doc-qa ask "What are the main findings?"

# Multi-hop question across documents (show each reasoning hop)
doc-qa ask "How do the methods in paper1 compare to paper2, and what datasets did each use?" --show-hops

# Restrict to specific documents
doc-qa ask "What was the accuracy reported?" --docs paper1.pdf --docs paper2.pdf

# Use a more powerful model with verbose token stats
doc-qa ask "Summarize the key contributions" --model claude-opus-4-6 --verbose
```

## All options

### `doc-qa index`

```
Arguments:
  PDF_FILES               One or more PDF files to index.

Options:
  --store PATH            JSON store file path. [default: .doc_qa_store.json]
  --chunk-size INT        Max tokens per chunk. [default: 400]
  --overwrite             Re-index already-indexed documents.
```

### `doc-qa ask`

```
Arguments:
  QUESTION                The question to answer.

Options:
  --store PATH            JSON store file path. [default: .doc_qa_store.json]
  -m, --model             Claude model to use. [default: claude-sonnet-4-6]
  --top-k INT             Chunks retrieved per sub-question. [default: 4]
  -d, --docs TEXT         Restrict to specific document filenames (repeatable).
  --max-sub-tokens INT    Max output tokens per sub-answer. [default: 512]
  --max-final-tokens INT  Max output tokens for final synthesis. [default: 1024]
  --show-hops             Print each sub-question and its individual answer.
  -v, --verbose           Show token usage after the answer.
```

## Multi-hop example

```
$ doc-qa ask "What problem does paper1 solve and how does paper2 build on it?" --show-hops

╔ Hop 1/2 ════════════════════════════════╗
│ Q: What problem does paper1 solve?      │
│ ...                                     │
╚═════════════════════════════════════════╝

╔ Hop 2/2 ════════════════════════════════╗
│ Q: How does paper2 build on paper1?     │
│ ...                                     │
╚═════════════════════════════════════════╝

╔ Multi-hop Answer ═══════════════════════╗
│ Paper1 addresses X by ... Paper2 then   │
│ extends this by ...                     │
╚═════════════════════════════════════════╝

Sources
 #  Document     Pages  Score  Excerpt
 1  paper1.pdf   3–4    8.42   The core problem addressed is…
 2  paper2.pdf   7      7.11   Building on prior work, we…
```

## Project structure

```
doc-qa/
├── pyproject.toml
└── doc_qa/
    ├── __init__.py
    ├── models.py      # Dataclasses: Chunk, Citation, SubAnswer, Answer
    ├── store.py       # PDF ingestion, chunking, JSON-backed persistence
    ├── retriever.py   # BM25 retrieval over indexed chunks
    ├── qa.py          # Multi-hop Q&A engine with citation tracking
    └── cli.py         # click CLI: index, list, ask commands
```
