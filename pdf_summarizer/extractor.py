"""PDF text extraction using pdfplumber."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


@dataclass
class Page:
    number: int  # 1-based
    text: str


@dataclass
class Document:
    path: Path
    pages: list[Page] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def num_pages(self) -> int:
        return len(self.pages)


def extract(pdf_path: str | Path) -> Document:
    """Extract text from a PDF file, returning a Document with per-page content."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    pages: list[Page] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(Page(number=i, text=text))

    return Document(path=path, pages=pages)
