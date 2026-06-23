from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from lxml import html
import psycopg
from psycopg import sql

from common import DEFAULT_FOLDER, TABLE_NAME, embed, ensure_schema, get_connection

SUPPORTED_SUFFIXES = {".doc", ".docx", ".html", ".htm", ".txt"}


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


def chunk_text(text: str, chunk_size: int = 1500, chunk_overlap: int = 200) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
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
            chunks.append(chunk)

        if end >= text_len:
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def ingest_file(
    conn: psycopg.Connection | None,
    file_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    dry_run: bool = False,
) -> int:
    print(f"Processing: {file_path}")

    text = load_document(file_path)
    if not text.strip():
        print(f"Skipped empty document: {file_path.name}")
        return 0

    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"  extracted_chars={len(text)} chunks={len(chunks)}")

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
                vector = embed(chunk)
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {table_name} (document_name, chunk_text, embedding)
                        VALUES (%s, %s, %s::vector)
                        """
                    ).format(table_name=sql.Identifier(TABLE_NAME)),
                    (document_name, chunk, vector),
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
