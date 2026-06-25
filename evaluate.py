from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from common import get_openai_client
from rag import build_context, build_messages, generate_answer, search_chunks

DEFAULT_TOP_K = int(os.getenv("EVAL_TOP_K", os.getenv("RAG_TOP_K", "5")))
DEFAULT_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", os.getenv("CHAT_MODEL", "gpt-4.1-mini"))
DEFAULT_OUTPUT_DIR = Path(os.getenv("EVAL_OUTPUT_DIR", "evals/results"))


@dataclass(slots=True)
class EvalCase:
    case_id: str
    question: str
    expected_documents: list[str]
    reference_answer: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class JudgeResult:
    verdict: str
    groundedness: int
    correctness: int | None
    completeness: int | None
    notes: str


WORD_RE = re.compile(r"\w+", re.UNICODE)


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def normalize_document_name(name: str) -> str:
    normalized = normalize_text(name)
    if "." in normalized:
        normalized = normalized.rsplit(".", 1)[0]
    return normalized


def document_matches(expected_name: str, retrieved_name: str) -> bool:
    expected = normalize_document_name(expected_name)
    retrieved = normalize_document_name(retrieved_name)
    return expected == retrieved or expected in retrieved or retrieved in expected


def ordered_unique_document_names(chunks) -> list[str]:
    names: list[str] = []
    for chunk in chunks:
        if chunk.document_name not in names:
            names.append(chunk.document_name)
    return names


def hit_at_k(retrieved_documents: list[str], expected_documents: list[str], k: int) -> int:
    top_docs = retrieved_documents[:k]
    return int(any(document_matches(expected, retrieved) for expected in expected_documents for retrieved in top_docs))


def recall_at_k(retrieved_documents: list[str], expected_documents: list[str], k: int) -> float | None:
    if not expected_documents:
        return None

    top_docs = retrieved_documents[:k]
    matches = 0
    for expected in expected_documents:
        if any(document_matches(expected, retrieved) for retrieved in top_docs):
            matches += 1
    return matches / len(expected_documents)


def mrr_at_k(retrieved_documents: list[str], expected_documents: list[str], k: int) -> float | None:
    if not expected_documents:
        return None

    for rank, retrieved in enumerate(retrieved_documents[:k], start=1):
        if any(document_matches(expected, retrieved) for expected in expected_documents):
            return 1.0 / rank
    return 0.0


def token_f1(reference_answer: str, candidate_answer: str) -> float | None:
    reference_tokens = WORD_RE.findall(reference_answer.lower())
    candidate_tokens = WORD_RE.findall(candidate_answer.lower())
    if not reference_tokens or not candidate_tokens:
        return None

    ref_counts: dict[str, int] = {}
    cand_counts: dict[str, int] = {}

    for token in reference_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    for token in candidate_tokens:
        cand_counts[token] = cand_counts.get(token, 0) + 1

    overlap = 0
    for token, ref_count in ref_counts.items():
        overlap += min(ref_count, cand_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find JSON object in judge response: {text!r}")
    return json.loads(text[start : end + 1])


def judge_answer(
    *,
    case: EvalCase,
    answer: str,
    retrieved_documents: list[str],
    context: str,
    judge_model: str,
) -> JudgeResult:
    client = get_openai_client()

    schema_hint = {
        "verdict": "pass or fail",
        "groundedness": "integer 1-5",
        "correctness": "integer 1-5 or null when no reference answer is provided",
        "completeness": "integer 1-5 or null when no reference answer is provided",
        "notes": "short explanation",
    }

    reference_answer = case.reference_answer or "[No reference answer provided]"
    prompt = (
        "You are evaluating a legal RAG answer. Return only one JSON object.\n"
        "Scoring rules:\n"
        "- groundedness: how well the answer stays within retrieved context.\n"
        "- correctness: how well the answer matches the reference answer. Use null if there is no reference answer.\n"
        "- completeness: how fully the answer covers the reference answer. Use null if there is no reference answer.\n"
        "- verdict: pass only if groundedness >= 4 and correctness/completeness are both >= 4 when reference answer exists.\n"
        f"Expected JSON schema: {json.dumps(schema_hint, ensure_ascii=False)}\n\n"
        f"Question:\n{case.question}\n\n"
        f"Expected documents:\n{json.dumps(case.expected_documents, ensure_ascii=False)}\n\n"
        f"Retrieved documents:\n{json.dumps(retrieved_documents, ensure_ascii=False)}\n\n"
        f"Retrieved context:\n{context or '[No context retrieved]'}\n\n"
        f"Reference answer:\n{reference_answer}\n\n"
        f"Candidate answer:\n{answer or '[No answer generated]'}"
    )

    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": "You are a strict evaluation judge. Output JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    payload = extract_json_object(response.choices[0].message.content or "{}")

    return JudgeResult(
        verdict=str(payload.get("verdict", "fail")).strip().lower(),
        groundedness=int(payload["groundedness"]),
        correctness=int(payload["correctness"]) if payload.get("correctness") is not None else None,
        completeness=int(payload["completeness"]) if payload.get("completeness") is not None else None,
        notes=str(payload.get("notes", "")).strip(),
    )


def parse_case_payload(payload: dict[str, Any], source_label: str, default_case_id: str) -> EvalCase:
    question = str(payload.get("question", "")).strip()
    if not question:
        raise ValueError(f"Missing question in {source_label}")

    case_id = str(payload.get("id") or default_case_id)
    expected_documents = payload.get("expected_documents") or []
    if not isinstance(expected_documents, list):
        raise ValueError(f"expected_documents must be a list in {source_label}")

    metadata = payload.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"metadata must be an object in {source_label}")

    reference_answer = payload.get("reference_answer")
    if reference_answer is not None:
        reference_answer = str(reference_answer).strip() or None

    return EvalCase(
        case_id=case_id,
        question=question,
        expected_documents=[str(item) for item in expected_documents],
        reference_answer=reference_answer,
        metadata=metadata,
    )


def load_cases(path: Path) -> list[EvalCase]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None
    else:
        if isinstance(parsed, list):
            return [
                parse_case_payload(item, f"{path} item {index}", f"case-{index}")
                for index, item in enumerate(parsed, start=1)
            ]
        if isinstance(parsed, dict):
            return [parse_case_payload(parsed, str(path), "case-1")]

    lines = path.read_text(encoding="utf-8").splitlines()
    jsonl_candidates = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if jsonl_candidates:
        try:
            return [
                parse_case_payload(json.loads(line), f"{path} line {index}", f"case-{index}")
                for index, line in enumerate(jsonl_candidates, start=1)
            ]
        except json.JSONDecodeError:
            pass

    cases: list[EvalCase] = []
    buffer: list[str] = []
    block_start_line = 1

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                if buffer:
                    raw = "\n".join(buffer)
                    payload = json.loads(raw)
                    cases.append(
                        parse_case_payload(payload, f"{path} block starting at line {block_start_line}", f"case-{len(cases) + 1}")
                    )
                    buffer = []
                continue

            if not buffer:
                block_start_line = line_number

            if stripped.startswith("#") and not buffer:
                continue

            buffer.append(line)

    if buffer:
        raw = "\n".join(buffer)
        payload = json.loads(raw)
        cases.append(
            parse_case_payload(payload, f"{path} block starting at line {block_start_line}", f"case-{len(cases) + 1}")
        )

    return cases


def summarize_numeric(values: list[float | None]) -> float | None:
    clean_values = [value for value in values if value is not None]
    if not clean_values:
        return None
    return mean(clean_values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer quality for the law RAG pipeline")
    parser.add_argument("dataset", type=Path, help="Path to a JSONL evaluation dataset")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="How many retrieved results to evaluate")
    parser.add_argument("--judge", action="store_true", help="Use an LLM judge for answer quality scoring")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Model used for LLM judging")
    parser.add_argument("--retrieval-only", action="store_true", help="Skip answer generation and run retrieval metrics only")
    parser.add_argument("--limit", type=int, help="Only evaluate the first N cases")
    parser.add_argument("--output", type=Path, help="Where to save the JSON result file")
    parser.add_argument("--min-hit-rate", type=float, help="Exit with code 1 if average hit@k falls below this value")
    parser.add_argument("--min-judge-pass-rate", type=float, help="Exit with code 1 if judge pass rate falls below this value")
    args = parser.parse_args()

    cases = load_cases(args.dataset)
    if args.limit is not None:
        cases = cases[: args.limit]

    if not cases:
        raise ValueError("No evaluation cases found")

    started_at = time.perf_counter()
    per_case_results: list[dict[str, Any]] = []

    for index, case in enumerate(cases, start=1):
        case_started_at = time.perf_counter()

        retrieval_started_at = time.perf_counter()
        chunks = search_chunks(case.question, top_k=args.top_k)
        retrieval_latency_ms = round((time.perf_counter() - retrieval_started_at) * 1000, 2)

        retrieved_documents = ordered_unique_document_names(chunks)
        context = build_context(chunks)

        answer_text: str | None = None
        answer_latency_ms: float | None = None
        judge_payload: dict[str, Any] | None = None

        if not args.retrieval_only:
            answer_started_at = time.perf_counter()
            messages = build_messages(case.question, chunks)
            answer_text = generate_answer(messages)
            answer_latency_ms = round((time.perf_counter() - answer_started_at) * 1000, 2)

            if args.judge:
                judge_result = judge_answer(
                    case=case,
                    answer=answer_text,
                    retrieved_documents=retrieved_documents,
                    context=context,
                    judge_model=args.judge_model,
                )
                judge_payload = asdict(judge_result)

        total_latency_ms = round((time.perf_counter() - case_started_at) * 1000, 2)
        ref_token_f1 = token_f1(case.reference_answer, answer_text) if case.reference_answer and answer_text else None

        result = {
            "id": case.case_id,
            "question": case.question,
            "expected_documents": case.expected_documents,
            "retrieved_documents": retrieved_documents,
            "retrieval": {
                "hit_at_k": hit_at_k(retrieved_documents, case.expected_documents, args.top_k),
                "recall_at_k": recall_at_k(retrieved_documents, case.expected_documents, args.top_k),
                "mrr_at_k": mrr_at_k(retrieved_documents, case.expected_documents, args.top_k),
                "top_chunk_similarity": chunks[0].similarity if chunks else None,
                "latency_ms": retrieval_latency_ms,
            },
            "answer": {
                "text": answer_text,
                "reference_answer": case.reference_answer,
                "token_f1": ref_token_f1,
                "latency_ms": answer_latency_ms,
                "judge": judge_payload,
            },
            "total_latency_ms": total_latency_ms,
            "metadata": case.metadata,
        }
        per_case_results.append(result)

        print(
            f"[{index}/{len(cases)}] {case.case_id} "
            f"hit@{args.top_k}={result['retrieval']['hit_at_k']} "
            f"mrr={result['retrieval']['mrr_at_k']} "
            f"latency_ms={total_latency_ms}"
        )

    total_runtime_ms = round((time.perf_counter() - started_at) * 1000, 2)

    hit_rates = [float(item["retrieval"]["hit_at_k"]) for item in per_case_results]
    recalls = [item["retrieval"]["recall_at_k"] for item in per_case_results]
    mrrs = [item["retrieval"]["mrr_at_k"] for item in per_case_results]
    answer_latencies = [item["answer"]["latency_ms"] for item in per_case_results]
    retrieval_latencies = [item["retrieval"]["latency_ms"] for item in per_case_results]
    total_latencies = [item["total_latency_ms"] for item in per_case_results]
    token_f1_values = [item["answer"]["token_f1"] for item in per_case_results]

    judge_results = [item["answer"]["judge"] for item in per_case_results if item["answer"]["judge"]]
    judge_pass_rate = None
    mean_groundedness = None
    mean_correctness = None
    mean_completeness = None
    if judge_results:
        judge_pass_rate = mean(1.0 if result["verdict"] == "pass" else 0.0 for result in judge_results)
        mean_groundedness = mean(result["groundedness"] for result in judge_results)
        correctness_values = [result["correctness"] for result in judge_results if result["correctness"] is not None]
        completeness_values = [result["completeness"] for result in judge_results if result["completeness"] is not None]
        mean_correctness = mean(correctness_values) if correctness_values else None
        mean_completeness = mean(completeness_values) if completeness_values else None

    summary = {
        "cases": len(per_case_results),
        "top_k": args.top_k,
        "retrieval": {
            "avg_hit_at_k": summarize_numeric(hit_rates),
            "avg_recall_at_k": summarize_numeric(recalls),
            "avg_mrr_at_k": summarize_numeric(mrrs),
            "avg_latency_ms": summarize_numeric(retrieval_latencies),
        },
        "answer": {
            "avg_token_f1": summarize_numeric(token_f1_values),
            "avg_latency_ms": summarize_numeric(answer_latencies),
            "judge_pass_rate": judge_pass_rate,
            "avg_groundedness": mean_groundedness,
            "avg_correctness": mean_correctness,
            "avg_completeness": mean_completeness,
        },
        "avg_total_latency_ms": summarize_numeric(total_latencies),
        "runtime_ms": total_runtime_ms,
    }

    output_payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "config": {
            "top_k": args.top_k,
            "judge": args.judge,
            "judge_model": args.judge_model if args.judge else None,
            "retrieval_only": args.retrieval_only,
        },
        "summary": summary,
        "results": per_case_results,
    }

    output_path = args.output
    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"eval-{timestamp}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSummary")
    print(f"- cases: {summary['cases']}")
    print(f"- avg hit@{args.top_k}: {summary['retrieval']['avg_hit_at_k']}")
    print(f"- avg recall@{args.top_k}: {summary['retrieval']['avg_recall_at_k']}")
    print(f"- avg mrr@{args.top_k}: {summary['retrieval']['avg_mrr_at_k']}")
    print(f"- avg retrieval latency ms: {summary['retrieval']['avg_latency_ms']}")
    if not args.retrieval_only:
        print(f"- avg answer latency ms: {summary['answer']['avg_latency_ms']}")
        print(f"- avg token f1: {summary['answer']['avg_token_f1']}")
        if args.judge:
            print(f"- judge pass rate: {summary['answer']['judge_pass_rate']}")
            print(f"- avg groundedness: {summary['answer']['avg_groundedness']}")
            print(f"- avg correctness: {summary['answer']['avg_correctness']}")
            print(f"- avg completeness: {summary['answer']['avg_completeness']}")
    print(f"- results saved to: {output_path}")

    failed = False
    if args.min_hit_rate is not None and (summary['retrieval']['avg_hit_at_k'] or 0.0) < args.min_hit_rate:
        print(f"ERROR: avg hit@{args.top_k} is below threshold {args.min_hit_rate}", file=sys.stderr)
        failed = True
    if args.min_judge_pass_rate is not None:
        actual_pass_rate = summary['answer']['judge_pass_rate']
        if actual_pass_rate is None or actual_pass_rate < args.min_judge_pass_rate:
            print(f"ERROR: judge pass rate is below threshold {args.min_judge_pass_rate}", file=sys.stderr)
            failed = True

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
