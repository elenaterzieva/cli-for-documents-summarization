# Architecture

## Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          pdf-summarizer CLI                         │
│                            (cli.py)                                 │
│                                                                     │
│   pdf-summarizer report.pdf --model claude-sonnet-4-6               │
│                    --strategy paragraph --chunk-size 1500           │
└────────────────────────────┬────────────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌─────────────┐  ┌───────────┐  ┌──────────────┐
     │  extractor  │  │  chunker  │  │  summarizer  │
     │ (extractor  │  │(chunker.py│  │(summarizer.py│
     │    .py)     │  │    )      │  │    )         │
     └──────┬──────┘  └─────┬─────┘  └──────┬───────┘
            │               │               │
            ▼               ▼               ▼
     pdfplumber         tiktoken       Anthropic API
     (page text)     (token count)    (Claude model)
```

## Data Flow

```
 PDF File
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ 1. EXTRACT                                           │
│    pdfplumber reads each page → list of Page objects │
│    Page { number, text }                             │
└───────────────────────────┬──────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────┐
│ 2. CHUNK                                             │
│                                                      │
│    Strategy: paragraph  ──► merge paragraphs         │
│              fixed      ──► split on token boundary  │
│              page       ──► group N pages together   │
│                                                      │
│    Output: list of Chunk objects                     │
│    Chunk { index, text, start_page, end_page }       │
└───────────────────────────┬──────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────┐
│ 3. SUMMARIZE (hierarchical)                          │
│                                                      │
│    For each chunk:                                   │
│      Claude API ──► chunk summary (max_chunk_tokens) │
│                                                      │
│    All chunk summaries combined:                     │
│      Claude API ──► final summary (max_final_tokens) │
└───────────────────────────┬──────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────┐
│ 4. OUTPUT                                            │
│    Terminal (rich panel) or --output file.md         │
│    Optional: --show-chunks, --verbose (token usage)  │
└──────────────────────────────────────────────────────┘
```

## Chunking Strategies

```
FIXED strategy (--chunk-size 1500 tokens)
─────────────────────────────────────────
 Full document text
 ├── Chunk 1 [tokens 0–1500]
 ├── Chunk 2 [tokens 1500–3000]
 └── Chunk 3 [tokens 3000–4200]


PARAGRAPH strategy (--chunk-size 1500 tokens)
──────────────────────────────────────────────
 Full document text split on blank lines
 ├── Para 1 (200 tok) ─┐
 ├── Para 2 (400 tok)  ├─► Chunk 1 (~1400 tok)
 ├── Para 3 (800 tok) ─┘
 ├── Para 4 (900 tok) ─┐
 └── Para 5 (500 tok) ─┴─► Chunk 2 (~1400 tok)


PAGE strategy (--pages-per-chunk 3)
─────────────────────────────────────
 ├── Pages 1–3  ──► Chunk 1
 ├── Pages 4–6  ──► Chunk 2
 └── Pages 7–9  ──► Chunk 3
```

## Hierarchical Summarization

```
        [Chunk 1]   [Chunk 2]   [Chunk 3]  ...  [Chunk N]
             │           │           │                │
             ▼           ▼           ▼                ▼
         [Sum 1]     [Sum 2]     [Sum 3]  ...     [Sum N]
             │           │           │                │
             └───────────┴───────────┴────────────────┘
                                 │
                                 ▼
                        [Final Summary]
```

For single-chunk documents, the chunk summary is returned directly as the final summary.

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `cli.py` | Parses CLI arguments, orchestrates the pipeline, renders output |
| `extractor.py` | Reads the PDF and returns `Document` with per-page `Page` objects |
| `chunker.py` | Splits a `Document` into `Chunk` objects using the chosen strategy |
| `summarizer.py` | Calls the Claude API per chunk then synthesizes a final summary |
