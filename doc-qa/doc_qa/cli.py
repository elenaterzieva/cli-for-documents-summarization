"""CLI entry point for doc-qa."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .qa import QAEngine
from .retriever import Retriever
from .store import DEFAULT_STORE, ChunkStore

console = Console()

MODELS = [
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-haiku-4-5-20251001",
]


@click.group()
def main() -> None:
    """Document Q&A with citation tracking and multi-hop question support."""


# ---------------------------------------------------------------------------
# index command
# ---------------------------------------------------------------------------

@main.command()
@click.argument("pdf_files", nargs=-1, required=True)
@click.option(
    "--store",
    default=str(DEFAULT_STORE),
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Path to the JSON chunk store file.",
)
@click.option(
    "--chunk-size",
    default=400,
    show_default=True,
    type=click.IntRange(50, 10000),
    help="Max tokens per chunk.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Re-index documents that are already in the store.",
)
def index(pdf_files: tuple[str, ...], store: str, chunk_size: int, overwrite: bool) -> None:
    """Index one or more PDF files into the chunk store.

    \b
    Examples:
      doc-qa index paper1.pdf paper2.pdf
      doc-qa index *.pdf --chunk-size 600 --overwrite
    """
    store_path = Path(store)
    chunk_store = ChunkStore(store_path)

    total_new = 0
    for pdf_file in pdf_files:
        path = Path(pdf_file)
        if not path.exists():
            console.print(f"[red]Not found:[/red] {pdf_file}")
            continue
        if path.suffix.lower() != ".pdf":
            console.print(f"[yellow]Skipping non-PDF:[/yellow] {pdf_file}")
            continue

        with console.status(f"[cyan]Indexing[/cyan] {path.name}..."):
            try:
                chunks = chunk_store.add_document(path, chunk_size=chunk_size, overwrite=overwrite)
                total_new += len(chunks)
                console.print(f"[green]Indexed[/green] {path.name} — {len(chunks)} chunks")
            except Exception as exc:
                console.print(f"[red]Failed to index {path.name}:[/red] {exc}")

    console.print(
        f"\n[bold]Store:[/bold] {store_path}  |  "
        f"[bold]Total chunks:[/bold] {chunk_store.chunk_count()}"
    )


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------

@main.command(name="list")
@click.option(
    "--store",
    default=str(DEFAULT_STORE),
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Path to the JSON chunk store file.",
)
def list_docs(store: str) -> None:
    """List all indexed documents in the store."""
    store_path = Path(store)
    if not store_path.exists():
        console.print("[yellow]No store found. Run `doc-qa index` first.[/yellow]")
        return

    chunk_store = ChunkStore(store_path)
    docs = chunk_store.indexed_docs()
    if not docs:
        console.print("[yellow]Store is empty.[/yellow]")
        return

    table = Table(title=f"Indexed documents ({store_path})", show_header=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Document")
    table.add_column("Path", style="dim")
    for i, entry in enumerate(docs, 1):
        name, _, path = entry.partition("  (")
        table.add_row(str(i), name.strip(), path.rstrip(")"))
    table.add_row("", f"[bold]{chunk_store.chunk_count()} total chunks[/bold]", "")
    console.print(table)


# ---------------------------------------------------------------------------
# ask command
# ---------------------------------------------------------------------------

@main.command()
@click.argument("question")
@click.option(
    "--store",
    default=str(DEFAULT_STORE),
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Path to the JSON chunk store file.",
)
@click.option(
    "--model",
    "-m",
    default="claude-sonnet-4-6",
    show_default=True,
    type=click.Choice(MODELS, case_sensitive=False),
    help="Claude model to use.",
)
@click.option(
    "--top-k",
    default=4,
    show_default=True,
    type=click.IntRange(1, 20),
    help="Number of chunks to retrieve per sub-question.",
)
@click.option(
    "--docs",
    "-d",
    multiple=True,
    help="Restrict search to these document filenames (can repeat). E.g. --docs paper1.pdf",
)
@click.option(
    "--max-sub-tokens",
    default=512,
    show_default=True,
    type=click.IntRange(64, 4096),
    help="Max output tokens per sub-answer.",
)
@click.option(
    "--max-final-tokens",
    default=1024,
    show_default=True,
    type=click.IntRange(128, 8192),
    help="Max output tokens for the final synthesis.",
)
@click.option(
    "--show-hops",
    is_flag=True,
    default=False,
    help="Show each sub-question and its answer.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show token usage and retrieval details.",
)
def ask(
    question: str,
    store: str,
    model: str,
    top_k: int,
    docs: tuple[str, ...],
    max_sub_tokens: int,
    max_final_tokens: int,
    show_hops: bool,
    verbose: bool,
) -> None:
    """Ask a question against indexed documents.

    \b
    Examples:
      doc-qa ask "What are the main findings?"
      doc-qa ask "Compare the methods in paper1 and paper2" --show-hops
      doc-qa ask "What datasets were used?" --docs paper1.pdf --top-k 6
    """
    store_path = Path(store)
    if not store_path.exists():
        console.print("[red]No store found. Run `doc-qa index` first.[/red]")
        sys.exit(1)

    chunk_store = ChunkStore(store_path)
    chunks = chunk_store.all_chunks()
    if not chunks:
        console.print("[red]Store is empty. Run `doc-qa index` first.[/red]")
        sys.exit(1)

    retriever = Retriever(chunks)
    engine = QAEngine(
        retriever=retriever,
        model=model,
        top_k=top_k,
        max_sub_tokens=max_sub_tokens,
        max_final_tokens=max_final_tokens,
    )

    doc_filter = list(docs) if docs else None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Thinking...", total=None)

        def _cb(msg: str) -> None:
            progress.update(task, description=msg)

        try:
            answer = engine.ask(question, doc_filter=doc_filter, progress=_cb)
        except Exception as exc:
            console.print(f"\n[red]Q&A failed:[/red] {exc}")
            sys.exit(1)

    # --- Render sub-answers (hops) ---
    if show_hops and answer.is_multi_hop:
        console.print()
        for i, sa in enumerate(answer.sub_answers):
            console.print(
                Panel(
                    f"[bold]Q:[/bold] {sa.sub_question}\n\n{sa.answer}",
                    title=f"[dim]Hop {i + 1}/{len(answer.sub_answers)}[/dim]",
                    border_style="dim",
                )
            )

    # --- Render final answer ---
    hop_label = "[cyan]Multi-hop[/cyan] " if answer.is_multi_hop else ""
    console.print(
        Panel(
            Markdown(answer.final_answer),
            title=f"{hop_label}[bold cyan]Answer[/bold cyan]",
            border_style="cyan",
        )
    )

    # --- Render citations ---
    if answer.citations:
        table = Table(title="Sources", show_header=True, box=None)
        table.add_column("", style="dim", width=3)
        table.add_column("Document")
        table.add_column("Pages")
        table.add_column("Score", justify="right", style="dim")
        table.add_column("Excerpt", style="dim")
        for i, c in enumerate(answer.citations, 1):
            pages = (
                str(c.start_page)
                if c.start_page == c.end_page
                else f"{c.start_page}–{c.end_page}"
            )
            table.add_row(
                str(i),
                c.doc_name,
                pages,
                f"{c.relevance_score:.2f}",
                c.excerpt[:60] + "…",
            )
        console.print(table)

    # --- Verbose token usage ---
    if verbose:
        vtable = Table(title="Token Usage", show_header=True)
        vtable.add_column("Metric", style="dim")
        vtable.add_column("Value", justify="right")
        vtable.add_row("Model", answer.model if hasattr(answer, "model") else model)
        vtable.add_row("Sub-questions", str(len(answer.sub_answers)))
        vtable.add_row("Citations", str(len(answer.citations)))
        vtable.add_row("Input tokens", f"{answer.total_input_tokens:,}")
        vtable.add_row("Output tokens", f"{answer.total_output_tokens:,}")
        vtable.add_row(
            "Total tokens",
            f"{answer.total_input_tokens + answer.total_output_tokens:,}",
        )
        console.print(vtable)


if __name__ == "__main__":
    main()
