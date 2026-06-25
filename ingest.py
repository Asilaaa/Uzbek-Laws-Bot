from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from lxml import html
import psycopg
from psycopg import sql

from common import DEFAULT_FOLDER, TABLE_NAME, embed, ensure_schema, get_connection

SUPPORTED_SUFFIXES = {".doc", ".docx", ".html", ".htm", ".txt"}
LAW_PDF_DIR = Path(os.getenv("LAW_PDF_DIR", str(Path.home() / "Documents" / "laws")))
PDFTOTEXT_BIN = shutil.which("pdftotext")
PDFINFO_BIN = shutil.which("pdfinfo")


@dataclass(slots=True)
class ChunkRecord:
    text: str
    page_start: int | None = None
    page_end: int | None = None


def normalize_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def load_docx(path: Path) -> str:
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs)
    return normalize_text(text)


def load_doc(path: Path) -> str:
    libreoffice = shutil.which("libreoffice")
    if libreoffice:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            subprocess.run(
                [
                    libreoffice,
                    "--headless",
                    "--convert-to",
                    "txt:Text",
                    "--outdir",
                    str(output_dir),
                    str(path),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            converted = output_dir / f"{path.stem}.txt"
            if converted.exists():
                return load_txt(converted)

    for command in (("antiword", str(path)), ("catdoc", str(path))):
        tool = shutil.which(command[0])
        if tool:
            result = subprocess.run(
                [tool, *command[1:]],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="ignore",
            )
            return normalize_text(result.stdout)

    raise RuntimeError(
        "Unable to read .doc files. Install libreoffice, antiword, or catdoc to enable .doc ingestion."
    )


def load_html_doc(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    tree = html.fromstring(raw)

    for bad in tree.xpath("//script|//style|//noscript|//head"):
        bad.drop_tree()

    blocks = []
    for node in tree.xpath("//p|//div|//li|//tr|//h1|//h2|//h3|//h4|//h5|//h6"):
        text = " ".join(part.strip() for part in node.itertext() if part.strip())
        if text:
            blocks.append(text)

    if not blocks:
        return normalize_text(tree.text_content())

    return normalize_text("\n".join(blocks))


def load_txt(path: Path) -> str:
    return normalize_text(path.read_text(encoding="utf-8", errors="ignore"))


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return load_docx(path)
    if suffix == ".doc":
        return load_doc(path)
    if suffix in {".html", ".htm"}:
        return load_html_doc(path)
    if suffix == ".txt":
        return load_txt(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def get_matching_pdf_path(file_path: Path) -> Path | None:
    pdf_path = LAW_PDF_DIR / f"{file_path.stem}.pdf"
    return pdf_path if pdf_path.exists() else None


def extract_pdf_page_count(pdf_path: Path) -> int:
    if not PDFINFO_BIN:
        raise RuntimeError("pdfinfo is required for page-aware citations. Install poppler-utils.")

    result = subprocess.run(
        [PDFINFO_BIN, str(pdf_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="ignore",
    )
    match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, flags=re.MULTILINE)
    if not match:
        raise RuntimeError(f"Unable to determine page count for PDF: {pdf_path}")
    return int(match.group(1))


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    if not PDFTOTEXT_BIN:
        raise RuntimeError("pdftotext is required for page-aware citations. Install poppler-utils.")

    page_count = extract_pdf_page_count(pdf_path)
    pages: list[tuple[int, str]] = []

    for page_number in range(1, page_count + 1):
        result = subprocess.run(
            [PDFTOTEXT_BIN, "-f", str(page_number), "-l", str(page_number), str(pdf_path), "-"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="ignore",
        )
        normalized = normalize_text(result.stdout)
        if normalized:
            pages.append((page_number, normalized))

    return pages


def chunk_text_with_spans(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> list[tuple[str, int, int]]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[tuple[str, int, int]] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            cut = max(
                text.rfind("\n", start + chunk_size // 2, end),
                text.rfind(" ", start + chunk_size // 2, end),
            )
            if cut > start:
                end = cut

        chunk = text[start:end].strip()
        if chunk:
            leading_trim = len(text[start:end]) - len(text[start:end].lstrip())
            trailing_trim = len(text[start:end]) - len(text[start:end].rstrip())
            chunk_start = start + leading_trim
            chunk_end = end - trailing_trim
            chunks.append((chunk, chunk_start, chunk_end))

        if end >= text_len:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> list[str]:
    return [chunk for chunk, _, _ in chunk_text_with_spans(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)]


def pages_for_span(span_start: int, span_end: int, page_spans: list[tuple[int, int, int]]) -> tuple[int | None, int | None]:
    matching_pages = [page_number for page_number, start, end in page_spans if not (span_end <= start or span_start >= end)]
    if not matching_pages:
        return None, None
    return matching_pages[0], matching_pages[-1]


def build_pdf_backed_chunks(file_path: Path, chunk_size: int, chunk_overlap: int) -> list[ChunkRecord]:
    pdf_path = get_matching_pdf_path(file_path)
    if pdf_path is None:
        return []

    pages = extract_pdf_pages(pdf_path)
    if not pages:
        return []

    combined_parts: list[str] = []
    page_spans: list[tuple[int, int, int]] = []
    cursor = 0

    for index, (page_number, page_text) in enumerate(pages):
        if index > 0:
            separator = "\n\n"
            combined_parts.append(separator)
            cursor += len(separator)

        combined_parts.append(page_text)
        start = cursor
        cursor += len(page_text)
        end = cursor
        page_spans.append((page_number, start, end))

    combined_text = "".join(combined_parts)
    chunk_spans = chunk_text_with_spans(combined_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    records: list[ChunkRecord] = []
    for chunk_text_value, span_start, span_end in chunk_spans:
        page_start, page_end = pages_for_span(span_start, span_end, page_spans)
        records.append(ChunkRecord(text=chunk_text_value, page_start=page_start, page_end=page_end))
    return records


def build_plain_chunks(file_path: Path, chunk_size: int, chunk_overlap: int) -> list[ChunkRecord]:
    text = load_document(file_path)
    return [ChunkRecord(text=chunk) for chunk in chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)]


def ingest_file(
    conn: psycopg.Connection | None,
    file_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    dry_run: bool = False,
) -> int:
    print(f"Processing: {file_path}")

    pdf_path = get_matching_pdf_path(file_path)
    chunks = build_pdf_backed_chunks(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap) if pdf_path else []

    if chunks:
        extracted_chars = sum(len(chunk.text) for chunk in chunks)
        print(f"  extracted_chars={extracted_chars} chunks={len(chunks)} page_backed=yes pdf={pdf_path.name}")
    else:
        text = load_document(file_path)
        if not text.strip():
            print(f"Skipped empty document: {file_path.name}")
            return 0
        chunks = build_plain_chunks(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(f"  extracted_chars={len(text)} chunks={len(chunks)} page_backed=no")

    if dry_run:
        return len(chunks)

    if conn is None:
        raise ValueError("A database connection is required when dry_run is False")

    document_name = file_path.name
    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {table_name} WHERE document_name = %s").format(
                    table_name=sql.Identifier(TABLE_NAME)
                ),
                (document_name,),
            )
            for chunk in chunks:
                vector = embed(chunk.text)
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {table_name} (document_name, chunk_text, page_start, page_end, embedding)
                        VALUES (%s, %s, %s, %s, %s::vector)
                        """
                    ).format(table_name=sql.Identifier(TABLE_NAME)),
                    (document_name, chunk.text, chunk.page_start, chunk.page_end, vector),
                )

    print(f"Done: {document_name}")
    return len(chunks)


def iter_supported_files(folder: Path, recursive: bool = False):
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    for path in sorted(iterator):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest legal documents into pgvector")
    parser.add_argument("--folder", type=Path, default=DEFAULT_FOLDER, help="Folder containing source documents")
    parser.add_argument("--chunk-size", type=int, default=1500)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--recursive", action="store_true", help="Recursively scan for supported files")
    parser.add_argument("--dry-run", action="store_true", help="Parse and chunk files without embedding/inserting")
    args = parser.parse_args()

    folder = args.folder
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = list(iter_supported_files(folder, recursive=args.recursive))
    if not files:
        print(f"No supported files found in: {folder}")
        return

    print(f"Found {len(files)} supported files in {folder}")

    conn = None
    try:
        if not args.dry_run:
            conn = get_connection()
            ensure_schema(conn)

        total_chunks = 0
        for file_path in files:
            total_chunks += ingest_file(
                conn=conn,
                file_path=file_path,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                dry_run=args.dry_run,
            )

        print(f"Finished. files={len(files)} total_chunks={total_chunks}")
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
