from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from minio import Minio
from psycopg import sql

from common import LAW_DOCUMENTS_TABLE, ensure_schema, get_connection

LAW_DOCS_DIR = Path(os.getenv("LAW_DOCS_DIR", str(Path.home() / "law-docs")))
LAW_PDF_DIR = Path(os.getenv("LAW_PDF_DIR", str(LAW_DOCS_DIR)))
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000").replace("http://", "").replace("https://", "")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").strip().lower() in {"1", "true", "yes", "on"}
MINIO_PUBLIC_BASE_URL = os.getenv("MINIO_PUBLIC_BASE_URL", "http://127.0.0.1:9000").rstrip("/")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "laws")
MINIO_PREFIX = os.getenv("MINIO_PREFIX", "raw").strip("/")
MINIO_PDF_PREFIX = os.getenv("MINIO_PDF_PREFIX", "pdf").strip("/")
SOURCE_EXTENSIONS = {".doc", ".docx", ".html", ".htm", ".txt"}


@dataclass(slots=True)
class SyncEntry:
    document_name: str
    upload_path: Path
    object_key: str
    source_url: str


def build_source_url(object_key: str) -> str:
    return f"{MINIO_PUBLIC_BASE_URL}/{MINIO_BUCKET}/{quote(object_key, safe='/')}"


def build_object_key(prefix: str, file_name: str) -> str:
    prefix = prefix.strip("/")
    return f"{prefix}/{file_name}" if prefix else file_name


def get_matching_pdf_path(document_path: Path) -> Path | None:
    pdf_name = f"{document_path.stem}.pdf"
    pdf_path = LAW_PDF_DIR / pdf_name
    return pdf_path if pdf_path.exists() else None


def build_sync_entry(document_path: Path, allow_raw_fallback: bool) -> SyncEntry | None:
    pdf_path = get_matching_pdf_path(document_path)
    if pdf_path is not None:
        object_key = build_object_key(MINIO_PDF_PREFIX, pdf_path.name)
        return SyncEntry(
            document_name=document_path.name,
            upload_path=pdf_path,
            object_key=object_key,
            source_url=build_source_url(object_key),
        )

    if allow_raw_fallback:
        object_key = build_object_key(MINIO_PREFIX, document_path.name)
        return SyncEntry(
            document_name=document_path.name,
            upload_path=document_path,
            object_key=object_key,
            source_url=build_source_url(object_key),
        )

    return None


def create_minio_client() -> Minio:
    if not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        raise ValueError("MINIO_ACCESS_KEY and MINIO_SECRET_KEY must be set")
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )


def ensure_bucket(client: Minio) -> None:
    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)


def upload_entry(client: Minio, entry: SyncEntry) -> None:
    content_type = "application/pdf" if entry.upload_path.suffix.lower() == ".pdf" else "application/octet-stream"
    client.fput_object(
        MINIO_BUCKET,
        entry.object_key,
        str(entry.upload_path),
        content_type=content_type,
    )


def iter_source_files() -> list[Path]:
    if not LAW_DOCS_DIR.exists() or not LAW_DOCS_DIR.is_dir():
        raise FileNotFoundError(f"Source directory not found: {LAW_DOCS_DIR}")
    return sorted(path for path in LAW_DOCS_DIR.iterdir() if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload source PDFs to MinIO and sync public URLs to law_documents")
    parser.add_argument(
        "--allow-raw-fallback",
        action="store_true",
        help="If a matching PDF is missing, upload the original source file instead.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned uploads and DB updates without changing anything")
    args = parser.parse_args()

    if not LAW_PDF_DIR.exists() or not LAW_PDF_DIR.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {LAW_PDF_DIR}")

    files = iter_source_files()
    if not files:
        print(f"No supported source files found in {LAW_DOCS_DIR}")
        return

    entries: list[SyncEntry] = []
    missing_pdf: list[str] = []

    for path in files:
        entry = build_sync_entry(path, allow_raw_fallback=args.allow_raw_fallback)
        if entry is None:
            missing_pdf.append(path.name)
            continue
        entries.append(entry)

    if missing_pdf:
        print("Skipped documents with no matching PDF:")
        for name in missing_pdf:
            print(f"  - {name}")

    if not entries:
        print("Nothing to sync")
        return

    for entry in entries:
        print(f"READY  {entry.document_name} -> {entry.object_key}")

    if args.dry_run:
        print(f"Dry run complete. ready={len(entries)} skipped={len(missing_pdf)}")
        return

    client = create_minio_client()
    ensure_bucket(client)

    for entry in entries:
        upload_entry(client, entry)
        print(f"UPLOADED  {entry.upload_path.name} -> {entry.object_key}")

    with get_connection() as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            for entry in entries:
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {table_name} (document_name, object_key, source_url)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (document_name) DO UPDATE
                        SET object_key = EXCLUDED.object_key,
                            source_url = EXCLUDED.source_url
                        """
                    ).format(table_name=sql.Identifier(LAW_DOCUMENTS_TABLE)),
                    (entry.document_name, entry.object_key, entry.source_url),
                )
        conn.commit()

    print(f"Synced {len(entries)} documents into {LAW_DOCUMENTS_TABLE}")


if __name__ == "__main__":
    main()
