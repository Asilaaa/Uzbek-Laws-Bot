# Evaluation process for this RAG project

This project already stores:
- retrieved-source-backed answers in `law_answers`
- user feedback in `law_feedback`

To make the system more production ready, add **offline evaluation** before changes go live and use **online feedback** after release.

## 1) Create a golden dataset

Use JSONL, one case per line.

Required fields:
- `question`: user-style question
- `expected_documents`: list of source document names you expect retrieval to surface

Optional fields:
- `id`: stable case id
- `reference_answer`: short ideal answer for answer-quality checks
- `metadata`: tags like topic, language, difficulty, source family

Example dataset:
- `evals/golden_set.example.jsonl`

Recommended first dataset size:
- 20-30 cases to start
- 50-100 cases for more reliable regression checks
- include easy, medium, hard, ambiguous, and no-answer cases

## 2) Run retrieval-only evaluation

```bash
python evaluate.py evals/golden_set.example.jsonl --retrieval-only --top-k 5
```

This measures:
- `hit@k`: whether any expected document appears in the top-k results
- `recall@k`: how many expected documents are retrieved
- `mrr@k`: how early the first relevant document appears
- retrieval latency

## 3) Run end-to-end answer evaluation

```bash
python evaluate.py evals/golden_set.example.jsonl --top-k 5
```

This adds:
- generated answer text
- answer latency
- token overlap F1 against `reference_answer` when available

## 4) Run LLM-as-judge evaluation

```bash
python evaluate.py evals/golden_set.example.jsonl --top-k 5 --judge
```

This adds judge scores for:
- `groundedness`
- `correctness`
- `completeness`
- `verdict`

Environment variables:
- `EVAL_JUDGE_MODEL` to override the judge model
- `EVAL_OUTPUT_DIR` to control where result JSON files are saved

## 5) Add release gates

You can fail CI if metrics drop below thresholds.

```bash
python evaluate.py evals/golden_set.example.jsonl \
  --top-k 5 \
  --judge \
  --min-hit-rate 0.85 \
  --min-judge-pass-rate 0.80
```

This exits with code `1` when thresholds are missed.

## 6) Use online evaluation too

Your Telegram bot already stores production signals:
- answers in `law_answers`
- thumbs up/down in `law_feedback`

Recommended weekly review:
- positive feedback rate
- questions with downvotes
- questions with no retrieved sources
- repeated unanswered topics
- latency trends

## 7) What good looks like

For a first production baseline, target roughly:
- high `hit@5` on curated cases
- strong groundedness from judge results
- stable answer latency
- improving thumbs-up rate over time

## 8) Practical next steps

1. Copy `evals/golden_set.example.jsonl` to a real dataset file.
2. Replace placeholder questions with real user queries.
3. Use exact `document_name` values from your database for `expected_documents`.
4. Add a few negative cases where the system should admit missing context.
5. Run evaluation before and after any ingestion, chunking, prompt, or model change.
