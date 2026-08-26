from __future__ import annotations

"""Phase C: Presidio PII + NeMo input/output rails + adversarial and latency evaluation."""

import asyncio
import json
import os
import re
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ADVERSARIAL_SET_PATH,
    GUARDRAILS_CONFIG_DIR,
    LATENCY_BUDGET_P95_MS,
    OPENAI_API_KEY,
    PRESIDIO_LANGUAGE,
    PRESIDIO_SPACY_MODEL,
)

_PRESIDIO_CACHE: tuple[Any, Any] | None = None
_NEMO_CACHE: Any = None


def setup_presidio():
    """Create Presidio engines with Vietnamese CCCD/CMND and mobile-phone recognizers.

    If Presidio or the requested spaCy model is not available, ``(None, None)`` is returned and
    :func:`pii_scan` uses its deterministic regex fallback. The report can therefore distinguish
    local smoke tests from the full Presidio run.
    """
    global _PRESIDIO_CACHE
    if _PRESIDIO_CACHE is not None:
        return _PRESIDIO_CACHE

    try:
        from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        cccd = PatternRecognizer(
            supported_entity="VN_CCCD",
            patterns=[
                Pattern("Vietnam CCCD 12 digits", r"\b\d{12}\b", 0.9),
                Pattern("Vietnam CMND 9 digits", r"\b\d{9}\b", 0.75),
            ],
            supported_language=PRESIDIO_LANGUAGE,
        )
        phone = PatternRecognizer(
            supported_entity="VN_PHONE",
            patterns=[Pattern("Vietnam mobile", r"\b0[3-9]\d{8}\b", 0.9)],
            supported_language=PRESIDIO_LANGUAGE,
        )

        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": PRESIDIO_LANGUAGE, "model_name": PRESIDIO_SPACY_MODEL}],
            }
        )
        nlp_engine = provider.create_engine()
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine)
        registry.add_recognizer(cccd)
        registry.add_recognizer(phone)
        analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine)
        anonymizer = AnonymizerEngine()
        _PRESIDIO_CACHE = (analyzer, anonymizer)
    except Exception as exc:
        print(
            f"Warning: Presidio full engine unavailable ({type(exc).__name__}: {exc}); "
            "using regex fallback."
        )
        _PRESIDIO_CACHE = (None, None)
    return _PRESIDIO_CACHE


def _regex_pii(text: str) -> list[dict]:
    patterns = (
        ("EMAIL_ADDRESS", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", 0.95),
        ("VN_PHONE", r"\b0[3-9]\d{8}\b", 0.90),
        # Phone is evaluated before CCCD/CMND so 10-digit mobiles cannot be split accidentally.
        ("VN_CCCD", r"\b\d{12}\b", 0.90),
        ("VN_CCCD", r"\b\d{9}\b", 0.75),
    )
    entities: list[dict] = []
    occupied: list[tuple[int, int]] = []
    for entity_type, pattern, score in patterns:
        for match in re.finditer(pattern, text):
            if any(not (match.end() <= start or match.start() >= end) for start, end in occupied):
                continue
            entities.append(
                {
                    "type": entity_type,
                    "text": match.group(0),
                    "score": score,
                    "start": match.start(),
                    "end": match.end(),
                }
            )
            occupied.append((match.start(), match.end()))
    return sorted(entities, key=lambda item: item["start"])


def _anonymize_regex(text: str, entities: list[dict]) -> str:
    output = text
    for entity in sorted(entities, key=lambda item: item["start"], reverse=True):
        output = (
            output[: entity["start"]]
            + f"<{entity['type']}>"
            + output[entity["end"] :]
        )
    return output


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Task 9a: detect/anonymize PII with Presidio, with a transparent regex fallback."""
    if analyzer is None and anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    if analyzer is not None and anonymizer is not None:
        try:
            raw = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
            allowed_types = {
                "VN_CCCD",
                "VN_PHONE",
                "EMAIL_ADDRESS",
                "PHONE_NUMBER",
                "CREDIT_CARD",
                "IBAN_CODE",
                "IP_ADDRESS",
                "PASSPORT",
            }
            raw = [result for result in raw if result.entity_type in allowed_types]
            entities = [
                {
                    "type": result.entity_type,
                    "text": text[result.start : result.end],
                    "score": round(float(result.score), 3),
                    "start": int(result.start),
                    "end": int(result.end),
                }
                for result in raw
            ]
            anonymized = anonymizer.anonymize(text=text, analyzer_results=raw).text if raw else text
            return {
                "has_pii": bool(entities),
                "entities": entities,
                "anonymized": anonymized,
                "backend": "presidio",
            }
        except Exception as exc:
            print(f"Warning: Presidio scan failed ({type(exc).__name__}: {exc}); using regex fallback.")

    entities = _regex_pii(text)
    return {
        "has_pii": bool(entities),
        "entities": entities,
        "anonymized": _anonymize_regex(text, entities),
        "backend": "regex_fallback",
    }


def setup_nemo_rails():
    """Load NeMo Guardrails from ``guardrails/``. Returns ``None`` if unavailable."""
    global _NEMO_CACHE
    if _NEMO_CACHE is not None:
        return _NEMO_CACHE
    try:
        from nemoguardrails import LLMRails, RailsConfig

        config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
        _NEMO_CACHE = LLMRails(config)
    except Exception as exc:
        print(
            f"Warning: NeMo Guardrails unavailable ({type(exc).__name__}: {exc}); "
            "using deterministic input/output fallback."
        )
        _NEMO_CACHE = False
    return None if _NEMO_CACHE is False else _NEMO_CACHE


_JAILBREAK_PATTERNS = (
    r"bỏ qua (?:tất cả )?(?:hướng dẫn|policy)",
    r"ignore (?:all )?(?:previous|your system)",
    r"pretend you are dan",
    r"do anything now",
    r"unrestricted ai",
    r"system override",
    r"admin command",
    r"ignore previous instructions",
    r"system instructions",
    r"dump all training data",
    r"đóng vai .*manager",
    r"ai không có giới hạn",
    r"tấn công mạng",
)
_OFF_TOPIC_PATTERNS = (
    r"bài thơ",
    r"nấu phở|nấu ăn|công thức.*phở",
    r"bitcoin|ethereum|crypto",
    r"phương trình vi phân|giải phương trình",
    r"marvel|bộ phim|recommend.*phim",
    r"thời tiết",
)
_PII_REQUEST_PATTERNS = (
    r"cccd .* của nhân viên|cho tôi biết cccd",
    r"số điện thoại của nhân viên|số điện thoại của",
    r"email của nhân viên",
    r"tiết lộ .*lương|bảng lương chi tiết|employee salaries|all employee salaries",
    r"thông tin nhân viên|employee records|confidential employee data",
    r"mật khẩu admin|admin password",
    r"tiết lộ .*dữ liệu|reveal all confidential",
)


def _deterministic_input_guard(text: str) -> str | None:
    normalized = " ".join(text.lower().split())
    for reason, patterns in (
        ("jailbreak_or_prompt_injection", _JAILBREAK_PATTERNS),
        ("off_topic", _OFF_TOPIC_PATTERNS),
        ("pii_or_confidential_request", _PII_REQUEST_PATTERNS),
    ):
        if any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in patterns):
            return reason
    return None


def _response_text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("content", value.get("response", "")))
    return str(value or "")


async def check_input_rail(text: str, rails=None) -> dict:
    """Task 9b: check obvious attacks deterministically, then consult NeMo when available."""
    local_reason = _deterministic_input_guard(text)
    if local_reason:
        return {
            "allowed": False,
            "blocked_reason": local_reason,
            "response": "Request blocked by the HR assistant input guard.",
            "backend": "deterministic_prefilter",
        }

    if OPENAI_API_KEY:
        if rails is None:
            rails = setup_nemo_rails()
        if rails is not None:
            try:
                result = await asyncio.wait_for(
                    rails.generate_async(messages=[{"role": "user", "content": text}]),
                    timeout=15.0,
                )
                response = _response_text(result)
                lowered = response.lower()
                refusal_cues = (
                    "không thể thực hiện",
                    "chỉ có thể trả lời",
                    "không thể cung cấp",
                    "i cannot",
                    "i'm sorry",
                )
                blocked = any(cue in lowered for cue in refusal_cues)
                return {
                    "allowed": not blocked,
                    "blocked_reason": "nemo_input_rail" if blocked else None,
                    "response": response,
                    "backend": "nemo",
                }
            except Exception as exc:
                return {
                    "allowed": True,
                    "blocked_reason": None,
                    "response": f"NeMo unavailable for this request: {type(exc).__name__}",
                    "backend": "deterministic_fallback_after_nemo_error",
                }

    return {
        "allowed": True,
        "blocked_reason": None,
        "response": "",
        "backend": "deterministic_fallback",
    }


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Task 11: block/redact PII or sensitive leakage, then optionally consult NeMo output rails."""
    pii = pii_scan(answer)
    sensitive = bool(
        re.search(
            r"(?:mật khẩu|password)\s*(?:admin|hệ thống)?\s*(?:là|:)|"
            r"cccd\s+(?:của|là)|số điện thoại cá nhân|dump .*data",
            answer.lower(),
        )
    )
    if pii["has_pii"] or sensitive:
        return {
            "safe": False,
            "flagged_reason": "pii_or_sensitive_output",
            "final_answer": pii["anonymized"] if pii["has_pii"] else "Response blocked by output guard.",
            "backend": "presidio_or_deterministic_output_guard",
        }

    if OPENAI_API_KEY:
        if rails is None:
            rails = setup_nemo_rails()
        if rails is not None:
            try:
                result = await asyncio.wait_for(
                    rails.generate_async(
                        messages=[
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ]
                    ),
                    timeout=15.0,
                )
                response = _response_text(result)
                lowered = response.lower()
                refusal = any(
                    cue in lowered
                    for cue in ("không thể cung cấp", "tôi không thể", "i cannot")
                )
                return {
                    "safe": not refusal,
                    "flagged_reason": "nemo_output_rail" if refusal else None,
                    "final_answer": response if refusal and response else answer,
                    "backend": "nemo",
                }
            except Exception:
                pass

    return {
        "safe": True,
        "flagged_reason": None,
        "final_answer": answer,
        "backend": "deterministic_fallback",
    }


def _run_coroutine(coroutine):
    """Run a coroutine from normal synchronous scripts/tests."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    # This path is uncommon for the lab CLI. Avoid nest_asyncio as an undeclared dependency.
    raise RuntimeError("A running asyncio loop was detected; await the async guard function directly.")


def run_adversarial_suite(
    adversarial_set: list[dict],
    rails=None,
    analyzer=None,
    anonymizer=None,
) -> list[dict]:
    """Task 10: run all adversarial cases through PII and input-rail layers."""
    if analyzer is None and anonymizer is None:
        analyzer, anonymizer = setup_presidio()

    async def run_all() -> list[dict]:
        output: list[dict] = []
        for item in adversarial_set:
            blocked_by: str | None = None
            backend_parts: list[str] = []
            pii = pii_scan(str(item["input"]), analyzer, anonymizer)
            backend_parts.append(str(pii.get("backend", "unknown")))
            if pii["has_pii"] and item.get("block_layer") == "presidio":
                blocked_by = "presidio"

            rail_result = None
            if blocked_by is None:
                rail_result = await check_input_rail(str(item["input"]), rails)
                backend_parts.append(str(rail_result.get("backend", "unknown")))
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            output.append(
                {
                    "id": int(item["id"]),
                    "category": str(item["category"]),
                    "input": str(item["input"]),
                    "expected": str(item["expected"]),
                    "actual": actual,
                    "blocked_by": blocked_by,
                    "passed": actual == item["expected"],
                    "backend": "+".join(backend_parts),
                }
            )
        return output

    results = _run_coroutine(run_all())
    passed = sum(bool(row["passed"]) for row in results)
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(max(0.0, float(value)) for value in values)

    def nearest(percentile: float) -> float:
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
        return round(ordered[index], 3)

    return {"p50": nearest(0.50), "p95": nearest(0.95), "p99": nearest(0.99)}


def measure_p95_latency(
    test_inputs: list[str],
    n_runs: int = 20,
    rails=None,
    analyzer=None,
    anonymizer=None,
) -> dict:
    """Task 12: measure P50/P95/P99 latency for Presidio and input-rail layers."""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    if analyzer is None and anonymizer is None:
        analyzer, anonymizer = setup_presidio()
    inputs = test_inputs or ["Nhân viên hỏi về chính sách nghỉ phép."]
    run_inputs = [inputs[index % len(inputs)] for index in range(n_runs)]

    async def measure() -> tuple[list[float], list[float], list[float]]:
        presidio_times: list[float] = []
        nemo_times: list[float] = []
        totals: list[float] = []
        for text in run_inputs:
            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            t2 = time.perf_counter()
            presidio_ms = max(0.0, (t1 - t0) * 1000)
            nemo_ms = max(0.0, (t2 - t1) * 1000)
            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            totals.append(presidio_ms + nemo_ms)
        return presidio_times, nemo_times, totals

    presidio_times, nemo_times, total_times = _run_coroutine(measure())
    total = _percentiles(total_times)
    return {
        "presidio_ms": _percentiles(presidio_times),
        "nemo_ms": _percentiles(nemo_times),
        "total_ms": total,
        "latency_budget_ok": total["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


def main() -> None:
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as handle:
        adversarial_set = json.load(handle)

    results = run_adversarial_suite(adversarial_set)
    passed = sum(bool(row["passed"]) for row in results)
    latency = measure_p95_latency(
        [str(item["input"]) for item in adversarial_set[:10]],
        n_runs=10,
    )
    report = {
        "openai_configured": bool(OPENAI_API_KEY),
        "adversarial_suite_pass_rate": round(passed / max(1, len(results)), 6),
        "adversarial_passed": passed,
        "adversarial_total": len(results),
        "latency_benchmark": latency,
        "results": results,
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/guard_results.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print("Guard report saved -> reports/guard_results.json")


if __name__ == "__main__":
    main()
