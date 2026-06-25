from __future__ import annotations

import argparse

from rag import search_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the law_chunks pgvector table")
    parser.add_argument("query", help="Natural language query to search for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of matches to return")
    parser.add_argument(
        "--document-name",
        help="Optional case-insensitive document name filter (substring match)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=500,
        help="Maximum number of chunk characters to print per result",
    )
    args = parser.parse_args()

    results = search_chunks(args.query, top_k=args.top_k, document_name=args.document_name)
    if not results:
        print("No matches found.")
        return

    for index, result in enumerate(results, start=1):
        preview = result.chunk_text[: args.max_chars].strip()
        if len(result.chunk_text) > args.max_chars:
            preview += "..."
        location = f"  {result.page_label}" if result.page_label else ""
        print(f"[{index}] {result.document_name}{location}  score={result.similarity:.4f}")
        print(preview)
        print()


if __name__ == "__main__":
    main()
