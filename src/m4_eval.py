from __future__ import annotations

"""M4 — RAG evaluation with RAGAS when available plus a deterministic offline proxy."""

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache

from config import (OPENAI_API_KEY, OPENAI_BASE_URL, RAGAS_EMBEDDING_DEVICE, RAGAS_EMBEDDING_MODEL,
                    RAGAS_MAX_WORKERS, RAGAS_MODEL, TEST_SET_PATH)

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

#: Rubric threshold for "a metric is healthy" — used by the diagnostic error tree.
PASS_THRESHOLD = 0.70


@dataclass
class EvalResult:
    """One evaluated row. A metric is ``None`` when the judge produced no measurement."""
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("test_set.json must contain a list")
    return data


def _score_or_none(value) -> float | None:
    """Clamp a RAGAS/numpy value into [0, 1], or None when it was not evaluated.

    RAGAS emits NaN when a judge call fails (rate limit, timeout, unparseable reply).
    NaN means "no measurement", **not** "scored zero" — folding it to 0.0 silently drags
    the aggregate down and manufactures failures that never happened. Callers must drop
    these rows from the mean and report how many were lost.
    """
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return max(0.0, min(1.0, score))


def _safe_score(value) -> float:
    """[0, 1] score for display/serialisation; unevaluated values render as 0.0."""
    score = _score_or_none(value)
    return 0.0 if score is None else score


def _validate_inputs(questions, answers, contexts, ground_truths) -> None:
    lengths = [len(questions), len(answers), len(contexts), len(ground_truths)]
    if len(set(lengths)) != 1:
        raise ValueError(
            "questions, answers, contexts and ground_truths must have equal length; "
            f"got {lengths}"
        )
    for index, item in enumerate(contexts):
        if not isinstance(item, (list, tuple)):
            raise TypeError(f"contexts[{index}] must be a list or tuple of strings")


# ─── Offline proxy metrics ────────────────────────────────────────────────


def _tokens(text: str) -> set[str]:
    stop = {"là", "và", "của", "có", "được", "cho", "trong", "một", "những", "các", "theo", "bao", "nhiêu"}
    return {x for x in re.findall(r"\w+", text.lower(), re.UNICODE) if len(x) > 1 and x not in stop}


def _overlap(a: str, b: str) -> float:
    aa, bb = _tokens(a), _tokens(b)
    if not aa:
        return 0.0
    return len(aa & bb) / len(aa)


def _offline_row(question: str, answer: str, contexts: list[str], ground_truth: str) -> EvalResult:
    ctx = "\n".join(contexts)
    faith = min(1.0, 0.55 * _overlap(answer, ctx) + 0.45 * _overlap(ground_truth, answer))
    relevancy = min(1.0, 0.45 * _overlap(question, answer) + 0.55 * _overlap(ground_truth, answer))
    if contexts:
        useful = [_overlap(ground_truth + " " + question, c) for c in contexts]
        precision = sum(1 for s in useful if s > 0.08) / len(useful)
        recall = max(useful)
    else:
        precision = recall = 0.0
    return EvalResult(question, answer, list(contexts), ground_truth, faith, relevancy, precision, recall)


def _aggregate(rows: list[EvalResult], backend: str, error: str | None = None) -> dict:
    """Mean each metric over the rows that were actually measured, never over NaNs."""
    out: dict = {}
    evaluated: dict[str, int] = {}
    for metric in METRIC_NAMES:
        vals = [v for v in (getattr(r, metric) for r in rows) if v is not None]
        evaluated[metric] = len(vals)
        out[metric] = sum(vals) / len(vals) if vals else 0.0
    out["per_question"] = rows
    out["backend"] = backend
    out["evaluated_counts"] = evaluated
    out["total_rows"] = len(rows)
    missing = {m: len(rows) - c for m, c in evaluated.items() if c < len(rows)}
    if missing:
        out["unevaluated_counts"] = missing
        note = ("judge returned no score for some rows (excluded from the mean, NOT counted "
                f"as zero): {missing}")
        out["error"] = f"{error}; {note}" if error else note
        print(f"  ⚠️  {note}")
    elif error:
        out["error"] = error
    return out


def evaluate_offline(questions, answers, contexts, ground_truths, error: str | None = None) -> dict:
    """Deterministic lexical proxy for the 4 RAGAS metrics — no API key required."""
    rows = [_offline_row(q, a, list(c), g) for q, a, c, g in zip(questions, answers, contexts, ground_truths)]
    return _aggregate(rows, "offline_proxy", error)


# ─── RAGAS LLM-judge backend ──────────────────────────────────────────────


@lru_cache(maxsize=1)
def _ragas_llm():
    """One deterministic judge per process, shared by baseline and production runs."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=RAGAS_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0,
        timeout=180,
        max_retries=2,
        tiktoken_model_name="gpt-4o-mini",
    )


@lru_cache(maxsize=1)
def _ragas_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=RAGAS_EMBEDDING_MODEL,
        model_kwargs={"device": RAGAS_EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def _evaluate_with_ragas(questions, answers, contexts, ground_truths) -> dict:
    """Run the real RAGAS 4-metric evaluation. Raises if the stack is unavailable."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    dataset = Dataset.from_dict({
        "question": [str(x or "").strip() for x in questions],
        "answer": [str(x or "").strip() for x in answers],
        "contexts": [[str(c).strip() for c in item if str(c).strip()] for item in contexts],
        "ground_truth": [str(x or "").strip() for x in ground_truths],
    })
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=_ragas_llm(),
        embeddings=_ragas_embeddings(),
        run_config=RunConfig(timeout=180, max_retries=3, max_wait=30, max_workers=RAGAS_MAX_WORKERS),
        raise_exceptions=False,
    )
    frame = result.to_pandas()
    rows: list[EvalResult] = []
    for _, row in frame.iterrows():
        row_contexts = row.get("contexts", [])
        if not isinstance(row_contexts, list):
            row_contexts = list(row_contexts) if row_contexts is not None else []
        rows.append(EvalResult(
            question=str(row.get("question", "")),
            answer=str(row.get("answer", "")),
            contexts=[str(x) for x in row_contexts],
            ground_truth=str(row.get("ground_truth", "")),
            faithfulness=_score_or_none(row.get("faithfulness")),
            answer_relevancy=_score_or_none(row.get("answer_relevancy")),
            context_precision=_score_or_none(row.get("context_precision")),
            context_recall=_score_or_none(row.get("context_recall")),
        ))
    return _aggregate(rows, "ragas")


def evaluate_ragas(questions: list[str], answers: list[str], contexts: list[list[str]],
                   ground_truths: list[str]) -> dict:
    """Evaluate 4 metrics.

    Prefers the real RAGAS LLM-judge when ``OPENAI_API_KEY`` is set and the RAGAS stack is
    importable. Falls back to a deterministic lexical proxy so the lab still produces a
    complete report offline; ``backend`` in the result records which path ran.
    """
    _validate_inputs(questions, answers, contexts, ground_truths)
    if not questions:
        return _aggregate([], "empty", "Evaluation dataset is empty")

    if not OPENAI_API_KEY:
        return evaluate_offline(questions, answers, contexts, ground_truths,
                                "OPENAI_API_KEY is not configured; used offline proxy metrics")
    try:
        return _evaluate_with_ragas(questions, answers, contexts, ground_truths)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"  ⚠️  RAGAS evaluation failed ({message}); falling back to offline proxy.")
        return evaluate_offline(questions, answers, contexts, ground_truths, message)


# ─── Failure analysis ─────────────────────────────────────────────────────


DIAGNOSTIC_TREE = {
    "faithfulness": (
        "Generation is not fully grounded in retrieved evidence.",
        "Use stricter context-only generation, keep temperature 0 and remove noisy chunks.",
    ),
    "answer_relevancy": (
        "Answer does not directly satisfy the question.",
        "Preserve intent, entities and requested units in the generation prompt.",
    ),
    "context_precision": (
        "Too many retrieved chunks are indirect or irrelevant.",
        "Improve reranking/metadata filtering or reduce final top-k.",
    ),
    "context_recall": (
        "Required evidence is missing from retrieved context.",
        "Improve chunking, hybrid recall, query decomposition or parent retrieval.",
    ),
}


def _error_tree(metrics: dict[str, float], worst: str) -> list[str]:
    """Walk the lecture's Error Tree top-down and record which branch failed."""
    return [
        f"Q1 Answer relevant to question? {metrics['answer_relevancy'] >= PASS_THRESHOLD} "
        f"({metrics['answer_relevancy']:.2f})",
        f"Q2 Context contains the evidence? {metrics['context_recall'] >= PASS_THRESHOLD} "
        f"({metrics['context_recall']:.2f})",
        f"Q3 Context free of noise? {metrics['context_precision'] >= PASS_THRESHOLD} "
        f"({metrics['context_precision']:.2f})",
        f"Q4 Answer grounded in context? {metrics['faithfulness'] >= PASS_THRESHOLD} "
        f"({metrics['faithfulness']:.2f})",
        f"→ Root branch: {worst}",
    ]


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    if bottom_n <= 0 or not eval_results:
        return []
    scored = []
    for item in eval_results:
        measured = {m: v for m in METRIC_NAMES if (v := _score_or_none(getattr(item, m))) is not None}
        if not measured:
            continue  # nothing was judged for this row; it is not evidence of a failure
        metrics = {m: _safe_score(getattr(item, m)) for m in METRIC_NAMES}
        avg = sum(measured.values()) / len(measured)
        worst = min(measured, key=measured.get)
        diagnosis, fix = DIAGNOSTIC_TREE[worst]
        q = item.question.lower()
        ctx = " ".join(item.contexts).lower()
        if any(x in q for x in ("hiện hành", "mới nhất", "hiện tại")) and "2023" in ctx and "2024" in ctx:
            diagnosis = "Conflicting policy versions were retrieved without a current-version preference."
            fix = "Extract version/effective-date metadata and boost/filter the current policy before generation."
        scored.append({
            "question": item.question,
            "answer": item.answer,
            "ground_truth": item.ground_truth,
            "contexts": list(item.contexts),
            "average_score": avg,
            "worst_metric": worst,
            "score": metrics[worst],
            "metrics": metrics,
            "unevaluated_metrics": [m for m in METRIC_NAMES if m not in measured],
            "diagnosis": diagnosis,
            "suggested_fix": fix,
            "error_tree": _error_tree(metrics, worst),
        })
    scored.sort(key=lambda x: (x["average_score"], x["score"]))
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    per = [asdict(x) if isinstance(x, EvalResult) else dict(x) for x in results.get("per_question", [])]
    payload = {
        "aggregate": {m: _safe_score(results.get(m, 0.0)) for m in METRIC_NAMES},
        "backend": results.get("backend", "unknown"),
        "num_questions": len(per),
        "evaluated_counts": results.get("evaluated_counts", {}),
        "unevaluated_counts": results.get("unevaluated_counts", {}),
        "configuration": {
            "judge_model": RAGAS_MODEL if results.get("backend") == "ragas" else None,
            "embedding_model": RAGAS_EMBEDDING_MODEL if results.get("backend") == "ragas" else None,
            "pass_threshold": PASS_THRESHOLD,
        },
        "per_question": per,
        "failures": failures,
    }
    if results.get("error"):
        payload["error"] = str(results["error"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run `python src/pipeline.py` to generate answers, then evaluate_ragas() runs automatically.")
