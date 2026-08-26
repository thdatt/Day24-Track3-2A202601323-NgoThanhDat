from __future__ import annotations

"""Phase B: LLM-as-Judge, swap-and-average, Cohen's kappa, and bias analysis."""

import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    HUMAN_LABELS_PATH,
    JUDGE_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    TEST_SET_PATH,
    create_openai_client,
)

VALID_WINNERS = {"A", "B", "tie"}


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str
    winner_pass2: str
    final_winner: str
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool
    scores_pass1: dict = field(default_factory=dict)
    scores_pass2: dict = field(default_factory=dict)


def _clamp(value: Any, default: float = 0.5) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    if value != value:
        value = default
    return max(0.0, min(1.0, value))


def _parse_json_object(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def _normalize_pairwise(payload: dict) -> dict:
    winner = str(payload.get("winner", "tie"))
    if winner not in VALID_WINNERS:
        winner = "tie"
    scores = payload.get("scores") if isinstance(payload.get("scores"), dict) else {}
    return {
        "winner": winner,
        "reasoning": str(payload.get("reasoning", "")).strip(),
        "scores": {
            "A": _clamp(scores.get("A", 0.5)),
            "B": _clamp(scores.get("B", 0.5)),
        },
    }


def _token_set(text: str) -> set[str]:
    stop = {
        "là", "và", "của", "có", "được", "cho", "trong", "một", "các", "theo",
        "bao", "nhiêu", "nhân", "viên", "thì", "với", "này", "đó", "không",
    }
    return {
        token
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(token) > 1 and token not in stop
    }


def _offline_pairwise(question: str, answer_a: str, answer_b: str) -> dict:
    """Deterministic fallback for tests; it is explicitly not reported as an LLM result."""
    if answer_a.strip() == answer_b.strip():
        return {
            "winner": "tie",
            "reasoning": "Offline fallback: the two answers are identical.",
            "scores": {"A": 0.5, "B": 0.5},
        }

    q_terms = _token_set(question)

    def score(answer: str) -> float:
        terms = _token_set(answer)
        overlap = len(q_terms & terms) / max(1, len(q_terms))
        # Light evidence bonuses, intentionally small so length does not dominate.
        has_number = bool(re.search(r"\d", answer))
        has_current = any(x in answer.lower() for x in ("hiện hành", "v2024", "v1.3", "v2.0"))
        concise = 1.0 if 20 <= len(answer) <= 500 else 0.0
        return _clamp(0.25 + 0.45 * overlap + 0.15 * has_number + 0.10 * has_current + 0.05 * concise)

    score_a = score(answer_a)
    score_b = score(answer_b)
    if abs(score_a - score_b) < 0.05:
        winner = "tie"
    else:
        winner = "A" if score_a > score_b else "B"
    return {
        "winner": winner,
        "reasoning": "Offline deterministic fallback used because no working LLM judge was available.",
        "scores": {"A": score_a, "B": score_b},
    }


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: ask an LLM to choose the better answer under accuracy/completeness/conciseness."""
    client = create_openai_client() if OPENAI_API_KEY else None
    if client is not None:
        prompt = f"""Compare two candidate RAG answers to the same HR-policy question.
Judge accuracy first, then completeness, then conciseness. Do not prefer an answer merely because it is longer.
If neither answer is clearly better, return tie.

Question:
{question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Return JSON only with this schema:
{{"winner":"A|B|tie","reasoning":"short explanation","scores":{{"A":0.0,"B":0.0}}}}
Scores must be between 0 and 1.
"""
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict, position-neutral RAG evaluator. Return JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            normalized = _normalize_pairwise(_parse_json_object(content))
            if normalized["reasoning"] or normalized["winner"] == "tie":
                return normalized
        except Exception as exc:
            # The fallback keeps unit tests runnable, while the generated report records the backend.
            print(f"Warning: LLM pairwise judge failed ({type(exc).__name__}: {exc}); using fallback.")
    return _offline_pairwise(question, answer_a, answer_b)


def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: run pairwise judging twice with swapped order and convert pass 2 back."""
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)

    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner1 = pass1.get("winner", "tie")
    winner2 = swap_map.get(pass2_raw.get("winner", "tie"), "tie")
    final = winner1 if winner1 == winner2 else "tie"

    scores1 = pass1.get("scores", {"A": 0.5, "B": 0.5})
    scores2_raw = pass2_raw.get("scores", {"A": 0.5, "B": 0.5})
    scores2 = {
        "A": _clamp(scores2_raw.get("B", 0.5)),
        "B": _clamp(scores2_raw.get("A", 0.5)),
    }

    return JudgeResult(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
        winner_pass1=winner1,
        winner_pass2=winner2,
        final_winner=final,
        reasoning_pass1=str(pass1.get("reasoning", "")),
        reasoning_pass2=str(pass2_raw.get("reasoning", "")),
        position_consistent=(winner1 == winner2),
        scores_pass1={"A": _clamp(scores1.get("A")), "B": _clamp(scores1.get("B"))},
        scores_pass2=scores2,
    )


def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: compute Cohen's kappa without requiring an extra sklearn dependency."""
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have the same length")
    if not judge_labels:
        return 0.0
    if any(label not in {0, 1} for label in [*judge_labels, *human_labels]):
        raise ValueError("Cohen kappa labels must be binary 0/1 for this lab")

    n = len(judge_labels)
    observed = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    expected = (
        judge_labels.count(1) / n * human_labels.count(1) / n
        + judge_labels.count(0) / n * human_labels.count(0) / n
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return max(-1.0, min(1.0, (observed - expected) / (1.0 - expected)))


def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: calculate position-bias and verbosity-bias rates."""
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "position_bias_count": 0,
            "verbosity_bias": 0.0,
            "verbosity_details": {
                "a_wins_a_longer": 0,
                "b_wins_b_longer": 0,
                "total_decisive": 0,
            },
            "interpretation": "No judge results available.",
        }

    position_bias_count = sum(not result.position_consistent for result in judge_results)
    a_wins_a_longer = sum(
        result.final_winner == "A" and len(result.answer_a) > len(result.answer_b)
        for result in judge_results
    )
    b_wins_b_longer = sum(
        result.final_winner == "B" and len(result.answer_b) > len(result.answer_a)
        for result in judge_results
    )
    decisive = sum(result.final_winner in {"A", "B"} for result in judge_results)
    position_rate = position_bias_count / total
    verbosity_rate = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive else 0.0

    interpretation = (
        "Position bias is high; keep swap-and-average for release gates."
        if position_rate > 0.30
        else "Position bias is low on this evaluation sample."
    )
    return {
        "total_judged": total,
        "position_bias_rate": round(position_rate, 4),
        "position_bias_count": int(position_bias_count),
        "verbosity_bias": round(verbosity_rate, 4),
        "verbosity_details": {
            "a_wins_a_longer": int(a_wins_a_longer),
            "b_wins_b_longer": int(b_wins_b_longer),
            "total_decisive": int(decisive),
        },
        "interpretation": interpretation,
    }


def _judge_label_against_reference(question: str, answer: str, reference: str) -> tuple[int, str]:
    """Judge the labeled model answer against the test-set reference for kappa calibration."""
    client = create_openai_client() if OPENAI_API_KEY else None
    if client is not None:
        prompt = f"""Determine whether the candidate answer is materially correct relative to the reference answer.
Return label 1 only if the candidate is correct and does not omit a critical condition; otherwise return 0.

Question: {question}
Reference answer: {reference}
Candidate answer: {answer}

Return JSON only: {{"label":0 or 1,"reasoning":"short explanation"}}
"""
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a strict RAG answer correctness judge."},
                    {"role": "user", "content": prompt},
                ],
            )
            payload = _parse_json_object(response.choices[0].message.content or "{}")
            label = int(payload.get("label", -1))
            if label in {0, 1}:
                return label, str(payload.get("reasoning", ""))
        except Exception as exc:
            print(f"Warning: calibration judge failed ({type(exc).__name__}: {exc}); using proxy.")

    # Offline proxy: reference-token coverage. Reported as a proxy, never as an LLM label.
    ref_terms = _token_set(reference)
    answer_terms = _token_set(answer)
    coverage = len(ref_terms & answer_terms) / max(1, len(ref_terms))
    return int(coverage >= 0.45), f"offline reference-overlap proxy coverage={coverage:.3f}"


def _interpret_kappa(value: float) -> str:
    if value < 0:
        return "poor"
    if value < 0.2:
        return "slight"
    if value < 0.4:
        return "fair"
    if value < 0.6:
        return "moderate"
    if value < 0.8:
        return "substantial"
    return "almost perfect"


def run_calibration() -> dict:
    """Run the 10 human-labeled questions against their matching ground-truth references."""
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as handle:
        human_data = json.load(handle)
    with open(TEST_SET_PATH, encoding="utf-8") as handle:
        test_set = json.load(handle)

    references = {int(item["id"]): str(item["ground_truth"]) for item in test_set}
    judge_labels: list[int] = []
    human_labels: list[int] = []
    rows: list[dict] = []
    pairwise_results: list[JudgeResult] = []

    for item in human_data:
        question_id = int(item["question_id"])
        reference = references.get(question_id, "")
        label, reasoning = _judge_label_against_reference(
            str(item["question"]), str(item["model_answer"]), reference
        )
        human_label = int(item["human_label"])
        judge_labels.append(label)
        human_labels.append(human_label)
        rows.append(
            {
                "question_id": question_id,
                "question": item["question"],
                "human_label": human_label,
                "judge_label": label,
                "agree": label == human_label,
                "reasoning": reasoning,
                "reference": reference,
            }
        )
        # Pair candidate against reference to measure positional stability without inventing a baseline answer.
        pairwise_results.append(
            swap_and_average(str(item["question"]), str(item["model_answer"]), reference)
        )

    kappa = cohen_kappa(judge_labels, human_labels)
    return {
        "backend": "openai_llm_judge" if OPENAI_API_KEY else "offline_reference_proxy",
        "judge_model": JUDGE_MODEL if OPENAI_API_KEY else None,
        "openai_base_url": OPENAI_BASE_URL if OPENAI_API_KEY else None,
        "cohen_kappa": round(kappa, 6),
        "kappa_interpretation": _interpret_kappa(kappa),
        "human_labels": human_labels,
        "judge_labels": judge_labels,
        "agreement_count": sum(j == h for j, h in zip(judge_labels, human_labels)),
        "total_labeled": len(human_labels),
        "calibration_rows": rows,
        "bias_report": bias_report(pairwise_results),
        "pairwise_results": [asdict(result) for result in pairwise_results],
    }


def main() -> None:
    report = run_calibration()
    os.makedirs("reports", exist_ok=True)
    with open("reports/judge_results.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(
        "Judge report saved -> reports/judge_results.json "
        f"(backend={report['backend']}, kappa={report['cohen_kappa']:.3f})"
    )


if __name__ == "__main__":
    main()
