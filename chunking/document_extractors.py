"""Document extractors for PDF and DOCX files."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

logger = logging.getLogger(__name__)


@dataclass
class DocumentBlock:
    content: str
    block_kind: str
    name: str
    tags: List[str]
    extra_metadata: dict


class PdfDocumentExtractor:
    """Extract page-oriented text blocks from PDF files."""

    def extract(self, file_path: str) -> List[DocumentBlock]:
        path = Path(file_path)
        blocks: List[DocumentBlock] = []

        try:
            for page_number, page in enumerate(self._iter_pages(file_path), start=1):
                text = (self._extract_page_text(page) or "").strip()
                ocr_used = False

                if self._should_attempt_ocr(text):
                    try:
                        ocr_text = (self._ocr_page_text(page) or "").strip()
                        if ocr_text:
                            text = ocr_text
                            ocr_used = True
                    except Exception as exc:
                        logger.warning("PDF OCR failed for %s page %d: %s", file_path, page_number, exc)

                if not text:
                    continue

                blocks.append(
                    DocumentBlock(
                        content=text,
                        block_kind="page",
                        name=f"{path.name}#page:{page_number}",
                        tags=["document", "pdf", "page"] + (["ocr"] if ocr_used else []),
                        extra_metadata={
                            "document_type": "pdf",
                            "block_kind": "page",
                            "page_number": page_number,
                            "ocr_used": ocr_used,
                            "source_format": path.suffix.lower(),
                        },
                    )
                )
        except Exception as exc:
            logger.warning("Failed to extract PDF %s: %s", file_path, exc)
            return []

        return blocks

    def _iter_pages(self, file_path: str) -> Iterable[object]:
        import pymupdf

        document = pymupdf.open(file_path)
        try:
            for page in document:
                yield page
        finally:
            document.close()

    def _extract_page_text(self, page: object) -> str:
        return page.get_text("text")

    def _ocr_page_text(self, page: object) -> str:
        languages = os.getenv("CODE_SEARCH_PDF_OCR_LANGUAGES", "eng").strip() or "eng"
        textpage = page.get_textpage_ocr(language=languages, dpi=150)
        return page.get_text("text", textpage=textpage)

    def _should_attempt_ocr(self, text: str) -> bool:
        if str(os.getenv("CODE_SEARCH_PDF_OCR", "0")).lower() not in {"1", "true", "yes"}:
            return False
        try:
            min_chars = max(0, int(os.getenv("CODE_SEARCH_PDF_OCR_MIN_TEXT_CHARS", "24")))
        except Exception:
            min_chars = 24
        return len((text or "").strip()) < min_chars


class DocxDocumentExtractor:
    """Extract section and table blocks from DOCX files."""

    def extract(self, file_path: str) -> List[DocumentBlock]:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        path = Path(file_path)
        document = Document(file_path)
        blocks: List[DocumentBlock] = []
        active_heading: str | None = None
        section_lines: List[str] = []
        section_index = 0
        table_index = 0

        def flush_section() -> None:
            nonlocal section_index, section_lines
            content = "\n".join(line for line in section_lines if line.strip()).strip()
            if not content:
                section_lines = []
                return
            section_index += 1
            if active_heading:
                content = f"{active_heading}\n\n{content}"
            blocks.append(
                DocumentBlock(
                    content=content,
                    block_kind="section",
                    name=f"{path.name}#section:{section_index}",
                    tags=["document", "docx", "section"],
                    extra_metadata={
                        "document_type": "docx",
                        "block_kind": "section",
                        "section_title": active_heading,
                        "source_format": path.suffix.lower(),
                    },
                )
            )
            section_lines = []

        for item in document.iter_inner_content():
            if isinstance(item, Paragraph):
                text = (item.text or "").strip()
                if not text:
                    continue
                style_name = getattr(getattr(item, "style", None), "name", "") or ""
                if style_name.startswith("Heading"):
                    flush_section()
                    active_heading = text
                    continue
                section_lines.append(text)
                continue

            if isinstance(item, Table):
                flush_section()
                rows = []
                for row in item.rows:
                    cells = [" ".join(cell.text.split()) for cell in row.cells]
                    rows.append(" | ".join(cells).strip())
                table_text = "\n".join(row for row in rows if row.strip()).strip()
                if not table_text:
                    continue
                table_index += 1
                if active_heading:
                    table_text = f"{active_heading}\n\n{table_text}"
                blocks.append(
                    DocumentBlock(
                        content=table_text,
                        block_kind="table",
                        name=f"{path.name}#table:{table_index}",
                        tags=["document", "docx", "table"],
                        extra_metadata={
                            "document_type": "docx",
                            "block_kind": "table",
                            "section_title": active_heading,
                            "source_format": path.suffix.lower(),
                        },
                    )
                )

        flush_section()
        return blocks
