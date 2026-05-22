"""CLI entry point for pdf-summarizer."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .chunker import Strategy, chunk_document
from .extractor import extract
from .summarizer import summarize

console = Console()

MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

STRATEGIES = [s.value for s in Strategy]


def _validate_pdf(ctx: click.Context, param: click.Parameter, value: str) -> str:
    path = Path(value)
    if not path.exists():
        raise click.BadParameter(f"File not found: {value}")
    if path.suffix.lower() != ".pdf":
        raise click.BadParameter(f"Expected a .pdf file, got: {path.suffix}")
    return value


@click.command()
@click.argument("pdf_file", callback=_validate_pdf)
@click.option(
    "--model",
    "-m",
    default="claude-sonnet-4-6",
    show_default=True,
    type=click.Choice(MODELS, case_sensitive=False),
    help="Claude model to use for summarization.",
)
@click.option(
    "--strategy",
    "-s",
    default="paragraph",
    show_default=True,
    type=click.Choice(STRATEGIES, case_sensitive=False),
    help=(
        "Chunking strategy:\n\n"
        "  fixed      — Split by exact token count.\n\n"
        "  paragraph  — Split on blank lines, up to chunk-size tokens.\n\n"
        "  page       — One chunk per N pages (see --pages-per-chunk)."
    ),
)
@click.option(
    "--chunk-size",
    "-c",
    default=1500,
    show_default=True,
    type=click.IntRange(100, 50000),
    help="Max tokens per chunk (used by 'fixed' and 'paragraph' strategies).",
)
@click.option(
    "--pages-per-chunk",
    default=3,
    show_default=True,
    type=click.IntRange(1, 100),
    help="Pages per chunk (used by 'page' strategy only).",
)
@click.option(
    "--max-chunk-tokens",
    default=512,
    show_default=True,
    type=click.IntRange(64, 4096),
    help="Max output tokens for each chunk summary.",
)
@click.option(
    "--max-final-tokens",
    default=1024,
    show_default=True,
    type=click.IntRange(128, 8192),
    help="Max output tokens for the final synthesis.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help="Write the final summary to this file instead of stdout.",
)
@click.option(
    "--show-chunks",
    is_flag=True,
    default=False,
    help="Also print individual chunk summaries.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show document stats and token usage.",
)
def main(
    pdf_file: str,
    model: str,
    strategy: str,
    chunk_size: int,
    pages_per_chunk: int,
    max_chunk_tokens: int,
    max_final_tokens: int,
    output: str | None,
    show_chunks: bool,
    verbose: bool,
) -> None:
    """Summarize a long PDF using Claude with configurable chunking strategies.

    \b
    Examples:
      pdf-summarizer report.pdf
      pdf-summarizer paper.pdf --model claude-opus-4-6 --strategy fixed --chunk-size 2000
      pdf-summarizer book.pdf --strategy page --pages-per-chunk 5 --output summary.md
    """
    pdf_path = Path(pdf_file)

    # --- Extract ---
    with console.status("[bold cyan]Extracting PDF text..."):
        try:
            doc = extract(pdf_path)
        except Exception as exc:
            console.print(f"[red]Extraction failed:[/red] {exc}")
            sys.exit(1)

    if verbose:
        console.print(
            f"[dim]Extracted {doc.num_pages} pages from[/dim] [bold]{pdf_path.name}[/bold]"
        )

    # --- Chunk ---
    with console.status("[bold cyan]Chunking document..."):
        chunks = chunk_document(
            doc,
            strategy=Strategy(strategy),
            chunk_size=chunk_size,
            pages_per_chunk=pages_per_chunk,
        )

    if not chunks:
        console.print("[red]No text found in PDF.[/red]")
        sys.exit(1)

    if verbose:
        console.print(
            f"[dim]Split into[/dim] [bold]{len(chunks)}[/bold] [dim]chunk(s) "
            f"using strategy=[/dim][bold]{strategy}[/bold]"
        )

    # --- Summarize ---
    completed = [0]  # mutable for closure

    def _progress(current: int, total: int, message: str) -> None:
        completed[0] = current

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Summarizing...", total=len(chunks) + 1)

        def _cb(current: int, total: int, message: str) -> None:
            progress.update(task, completed=current, description=message)

        try:
            result = summarize(
                chunks=chunks,
                model=model,
                title=pdf_path.stem,
                strategy=strategy,
                chunk_size=chunk_size,
                max_chunk_tokens=max_chunk_tokens,
                max_final_tokens=max_final_tokens,
                progress_callback=_cb,
            )
        except Exception as exc:
            console.print(f"\n[red]Summarization failed:[/red] {exc}")
            sys.exit(1)

    # --- Output ---
    if show_chunks:
        for i, (chunk, summary) in enumerate(zip(chunks, result.chunk_summaries)):
            pages = (
                f"p{chunk.start_page}"
                if chunk.start_page == chunk.end_page
                else f"p{chunk.start_page}–{chunk.end_page}"
            )
            console.print(
                Panel(
                    summary,
                    title=f"[bold]Chunk {i + 1}[/bold] [dim]({pages})[/dim]",
                    border_style="dim",
                )
            )

    if output:
        out_path = Path(output)
        out_path.write_text(result.final_summary, encoding="utf-8")
        console.print(f"\n[green]Summary written to[/green] [bold]{out_path}[/bold]")
    else:
        console.print(
            Panel(
                Markdown(result.final_summary),
                title=f"[bold cyan]Summary — {pdf_path.name}[/bold cyan]",
                border_style="cyan",
            )
        )

    if verbose:
        table = Table(title="Token Usage", show_header=True)
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        table.add_row("Model", result.model)
        table.add_row("Strategy", result.strategy)
        table.add_row("Chunks", str(result.num_chunks))
        table.add_row("Input tokens", f"{result.total_input_tokens:,}")
        table.add_row("Output tokens", f"{result.total_output_tokens:,}")
        table.add_row(
            "Total tokens",
            f"{result.total_input_tokens + result.total_output_tokens:,}",
        )
        console.print(table)


if __name__ == "__main__":
    main()
