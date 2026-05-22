# CLI for Documents Summarization

A command-line tool that summarizes long PDF documents using the Anthropic Claude API. It supports configurable chunking strategies and models, making it suitable for everything from short reports to book-length documents.

## How it works

Because large PDFs exceed model context windows, the tool uses **hierarchical summarization**:

1. The PDF is extracted page-by-page using `pdfplumber`.
2. The text is split into chunks using one of three strategies.
3. Each chunk is summarized individually by Claude.
4. All chunk summaries are synthesized into a single final summary.

## Chunking strategies

| Strategy | Flag | Description |
|---|---|---|
| `paragraph` | `--strategy paragraph` | Merges paragraphs (blank-line separated) up to the token limit. Best for prose. |
| `fixed` | `--strategy fixed` | Splits on exact token count boundaries. Simple and predictable. |
| `page` | `--strategy page` | Groups N pages per chunk. Good for structured documents. |

## Supported models

- `claude-sonnet-4-6` (default)
- `claude-opus-4-6`
- `claude-haiku-4-5-20251001`

## Installation

Requires Python 3.10+.

```bash
cd pdf-summarizer
pip install -e .
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_key_here   # macOS/Linux
set ANTHROPIC_API_KEY=your_key_here      # Windows
```

## Usage

```bash
# Basic — uses paragraph strategy and claude-sonnet-4-6 by default
pdf-summarizer report.pdf

# Use a specific model and chunking strategy
pdf-summarizer paper.pdf --model claude-opus-4-6 --strategy fixed --chunk-size 2000

# Page-based chunking, 5 pages per chunk, save output to a file
pdf-summarizer book.pdf --strategy page --pages-per-chunk 5 --output summary.md

# Show per-chunk summaries and token usage stats
pdf-summarizer report.pdf --show-chunks --verbose
```

## All options

```
Arguments:
  PDF_FILE                    Path to the PDF file to summarize.

Options:
  -m, --model                 Claude model to use. [default: claude-sonnet-4-6]
  -s, --strategy              Chunking strategy: fixed | paragraph | page. [default: paragraph]
  -c, --chunk-size            Max tokens per chunk (fixed/paragraph). [default: 1500]
  --pages-per-chunk           Pages per chunk (page strategy only). [default: 3]
  --max-chunk-tokens          Max output tokens per chunk summary. [default: 512]
  --max-final-tokens          Max output tokens for the final summary. [default: 1024]
  -o, --output PATH           Write final summary to a file instead of stdout.
  --show-chunks               Also print individual chunk summaries.
  -v, --verbose               Show document stats and token usage after completion.
  --help                      Show this message and exit.
```

## Project structure

```
pdf-summarizer/
├── pyproject.toml
└── pdf_summarizer/
    ├── __init__.py
    ├── extractor.py   # PDF text extraction (pdfplumber)
    ├── chunker.py     # Chunking strategies (fixed, paragraph, page)
    ├── summarizer.py  # Hierarchical summarization via Claude API
    └── cli.py         # click-based CLI entry point
```

## Testing manually

After installing, run against any PDF:

```bash
# Verbose output shows page count, chunk count, and token usage
pdf-summarizer sample.pdf --verbose

# Inspect how the document is chunked before reviewing the final summary
pdf-summarizer sample.pdf --show-chunks --strategy fixed --chunk-size 1000
```

To test without a real PDF, create a minimal one with any word processor and export it as PDF.
