# Architecture

## Component Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          doc-qa CLI                                  │
│                          (cli.py)                                    │
│                                                                      │
│   doc-qa index paper1.pdf paper2.pdf                                 │
│   doc-qa ask "How do these papers compare?" --show-hops              │
└──────────────┬──────────────────────────────┬───────────────────────┘
               │                              │
     ┌─────────▼──────────┐       ┌──────────▼──────────┐
     │   ChunkStore       │       │    QAEngine          │
     │   (store.py)       │       │    (qa.py)           │
     │                    │       │                      │
     │ • PDF extraction   │       │ • Decompose question │
     │ • Paragraph chunk  │       │ • Retrieve per hop   │
     │ • JSON persistence │       │ • Synthesize answer  │
     └─────────┬──────────┘       └──────────┬──────────┘
               │                             │
     ┌─────────▼──────────┐       ┌──────────▼──────────┐
     │   pdfplumber       │       │   Retriever          │
     │   tiktoken         │       │   (retriever.py)     │
     └────────────────────┘       │                      │
                                  │ • BM25 index         │
                                  │ • top-k retrieval    │
                                  │ • Citation mapping   │
                                  └──────────┬──────────┘
                                             │
                                  ┌──────────▼──────────┐
                                  │   Anthropic API     │
                                  │   (Claude model)    │
                                  └─────────────────────┘
```

## Data Flow

```
 PDF Files
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ 1. INDEX (doc-qa index)                              │
│                                                      │
│    pdfplumber → page text                            │
│    tiktoken   → token-bounded paragraph chunks       │
│    JSON store → .doc_qa_store.json                   │
│                                                      │
│    Chunk { id, doc_name, start_page, end_page, text }│
└───────────────────────────┬──────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────┐
│ 2. DECOMPOSE (doc-qa ask)                            │
│                                                      │
│    Claude: "Is this multi-hop? Break it down."       │
│                                                      │
│    Single-hop:  ["What is X?"]                       │
│    Multi-hop:   ["What is X?", "How does Y use X?"]  │
└───────────────────────────┬──────────────────────────┘
                            │
                   for each sub-question
                            │
                            ▼
┌──────────────────────────────────────────────────────┐
│ 3. RETRIEVE                                          │
│                                                      │
│    BM25 over all chunks → top-k most relevant        │
│    Optional: --docs filter limits to specific files  │
└───────────────────────────┬──────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────┐
│ 4. SUB-ANSWER                                        │
│                                                      │
│    Claude answers each sub-question using only       │
│    the retrieved chunk context                       │
│    Citations tracked per hop                         │
└───────────────────────────┬──────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────┐
│ 5. SYNTHESIZE                                        │
│                                                      │
│    Claude combines all sub-answers into one          │
│    coherent final answer                             │
│    (skipped for single-hop questions)                │
└───────────────────────────┬──────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────┐
│ 6. OUTPUT                                            │
│                                                      │
│    Final answer (rich panel + markdown)              │
│    Citations table (doc, pages, score, excerpt)      │
│    Optional: --show-hops, --verbose                  │
└──────────────────────────────────────────────────────┘
```

## Multi-hop Reasoning

```
User question: "How do paper1 and paper2 differ in methodology?"
        │
        ▼
  ┌─────────────────────────────────────┐
  │  DECOMPOSE (Claude)                 │
  └──────┬──────────────────────┬───────┘
         │                      │
         ▼                      ▼
  "What methodology     "What methodology
   does paper1 use?"     does paper2 use?"
         │                      │
    BM25 retrieve          BM25 retrieve
    paper1 chunks          paper2 chunks
         │                      │
    Claude answers         Claude answers
    sub-question 1         sub-question 2
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
          ┌─────────────────┐
          │  SYNTHESIZE     │
          │  (Claude)       │
          └────────┬────────┘
                   │
                   ▼
          Final unified answer
          + deduplicated citations
          from both documents
```

## Citation Tracking

```
Each retrieved chunk becomes a Citation:

  Chunk { id, doc_name, start_page, end_page, text }
       │
       ▼
  Citation {
    chunk_id        → links back to source chunk
    doc_name        → "paper1.pdf"
    start_page      → 3
    end_page        → 4
    relevance_score → 8.42  (BM25 score)
    excerpt         → first 200 chars of chunk text
  }

Citations are deduplicated across hops and sorted by relevance score.
```

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `models.py` | Dataclasses: `Chunk`, `Citation`, `SubAnswer`, `Answer` |
| `store.py` | PDF ingestion, paragraph chunking, JSON-backed persistence |
| `retriever.py` | BM25 index construction, top-k retrieval, citation mapping |
| `qa.py` | Question decomposition, per-hop retrieval+answering, final synthesis |
| `cli.py` | `index`, `list`, `ask` commands with rich terminal output |
