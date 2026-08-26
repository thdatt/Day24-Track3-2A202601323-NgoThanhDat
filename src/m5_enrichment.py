from __future__ import annotations

"""M5 — chunk enrichment: summary, HyQA, contextual prepend and metadata.

Two modes, per the assignment:

* ``combined`` (default, bonus) — :func:`_enrich_single_call` produces summary + hypothesis
  questions + contextual sentence + metadata in **one** LLM call per chunk.
* individual — :func:`summarize_chunk`, :func:`generate_hypothesis_questions`,
  :func:`contextual_prepend` and :func:`extract_metadata` called separately (4 calls/chunk).

Every LLM path has a deterministic offline fallback so the lab runs without an API key.
"""

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from config import (ENRICH_CACHE_ENABLED, ENRICH_INDIVIDUAL_LLM, ENRICH_MAX_WORKERS, OPENAI_API_KEY,
                    OPENAI_MODEL, ROOT, create_openai_client)

#: Bump when the enrichment prompt changes so stale cache entries are ignored.
PROMPT_VERSION = "v1"
CACHE_PATH = str(ROOT / ".cache" / "enrichment.json")


@dataclass
class EnrichedChunk:
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str


#: Populated by :func:`enrich_chunks`; consumed by the latency report.
LAST_ENRICHMENT_STATS: dict[str, float | int] = {}


def _sentences(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if x.strip()]


def _fallback_metadata(text: str) -> dict:
    lower = text.lower()
    topics = {
        "leave": ["nghỉ", "phép", "ốm"],
        "security": ["mật khẩu", "mfa", "vpn"],
        "finance": ["tạm ứng", "thanh toán", "chi phí"],
        "salary": ["lương", "senior", "junior"],
        "training": ["đào tạo", "khóa học"],
        "safety": ["an toàn", "pccc", "sơ cứu"],
    }
    topic = "general"
    for k, words in topics.items():
        if any(w in lower for w in words):
            topic = k
            break
    years = re.findall(r"20\d{2}", text)
    amounts = re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:triệu|nghìn|%|ngày|tháng|VNĐ)?", text, flags=re.I)
    return {"topic": topic, "years": years, "numeric_facts": amounts[:8]}


def _fallback_summary(text: str) -> str:
    s = _sentences(text)
    return " ".join(s[:2]) if s else text.strip()


def _fallback_questions(text: str, n_questions: int = 3) -> list[str]:
    topic = _fallback_metadata(text)["topic"]
    labels = {
        "leave": "nghỉ phép", "security": "chính sách bảo mật", "finance": "quy định tạm ứng/thanh toán",
        "salary": "khung lương", "training": "đào tạo", "safety": "an toàn", "general": "nội dung tài liệu",
    }
    t = labels[topic]
    qs = [
        f"Tài liệu quy định gì về {t}?",
        f"Điều kiện hoặc thời hạn của {t} là bao nhiêu?",
        f"Ai áp dụng và cần làm gì theo quy định {t}?",
    ]
    return qs[:n_questions]


# ─── Individual techniques (4 calls/chunk mode) ───────────────────────────


def summarize_chunk(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if OPENAI_API_KEY and ENRICH_INDIVIDUAL_LLM:
        try:
            client = create_openai_client()
            resp = client.chat.completions.create(
                model=OPENAI_MODEL, temperature=0, max_tokens=120,
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn sau trong tối đa 2 câu tiếng Việt, chỉ dùng thông tin trong đoạn."},
                    {"role": "user", "content": text},
                ])
            return resp.choices[0].message.content.strip()
        except Exception:
            pass
    return _fallback_summary(text)


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    if n_questions <= 0 or not text.strip():
        return []
    if OPENAI_API_KEY and ENRICH_INDIVIDUAL_LLM:
        try:
            client = create_openai_client()
            resp = client.chat.completions.create(
                model=OPENAI_MODEL, temperature=0, max_tokens=200,
                messages=[
                    {"role": "system", "content": f"Sinh đúng {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời. Mỗi câu một dòng, không đánh số."},
                    {"role": "user", "content": text},
                ])
            lines = [re.sub(r"^\s*[-*\d.)]+\s*", "", x).strip()
                     for x in resp.choices[0].message.content.splitlines() if x.strip()]
            if lines:
                return lines[:n_questions]
        except Exception:
            pass
    return _fallback_questions(text, n_questions)


def contextual_prepend(text: str, document_title: str = "") -> str:
    text = text.strip()
    if not text:
        return ""
    context = f"Trích từ tài liệu {document_title}." if document_title else "Trích từ tài liệu nội bộ."
    return f"[Bối cảnh] {context}\n\n{text}"


def extract_metadata(text: str) -> dict:
    if OPENAI_API_KEY and ENRICH_INDIVIDUAL_LLM:
        try:
            client = create_openai_client()
            resp = client.chat.completions.create(
                model=OPENAI_MODEL, temperature=0, max_tokens=200,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": 'Trả về JSON {"topic":..., "entities":[...], "category":"hr|it|finance|compliance|safety|general"} dựa trên đoạn văn.'},
                    {"role": "user", "content": text},
                ])
            parsed = _parse_json_object(resp.choices[0].message.content)
            if parsed:
                return {**_fallback_metadata(text), **parsed}
        except Exception:
            pass
    return _fallback_metadata(text)


# ─── Combined single-call mode (1 call/chunk) ─────────────────────────────


_SYSTEM_PROMPT = """Bạn là bộ phân tích chunk cho hệ thống RAG tiếng Việt.
Chỉ dùng sự thật có trong tên tài liệu và đoạn văn, không suy diễn.
Trả về một JSON object đúng schema:
{
  "summary": "tóm tắt tối đa 2 câu",
  "questions": ["3 câu hỏi mà đoạn có thể trả lời"],
  "context": "1 câu nói chunk thuộc tài liệu nào và chủ đề gì",
  "metadata": {
    "topic": "chủ đề",
    "entities": ["thực thể"],
    "category": "hr|it|finance|compliance|safety|general",
    "version": "phiên bản hoặc null",
    "effective_date": "ngày hiệu lực hoặc null"
  }
}
Không dùng Markdown code fence."""


def _parse_json_object(raw: str | None) -> dict:
    if not raw:
        return {}
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    try:
        parsed = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            return {}
    return parsed if isinstance(parsed, dict) else {}


def _normalize_result(parsed: dict, text: str, source: str) -> dict:
    """Coerce any LLM payload into the fixed enrichment shape, filling gaps from fallbacks."""
    summary = str(parsed.get("summary") or "").strip() or _fallback_summary(text)
    raw_questions = parsed.get("questions") or []
    if isinstance(raw_questions, str):
        raw_questions = [raw_questions]
    questions = [str(q).strip() for q in raw_questions if str(q).strip()][:3] or _fallback_questions(text, 3)
    default_context = f"Trích từ tài liệu {source}." if source else "Trích từ tài liệu nội bộ."
    context = str(parsed.get("context") or "").strip() or default_context
    metadata = parsed.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    return {
        "summary": summary,
        "questions": questions,
        "context": context,
        "metadata": {**_fallback_metadata(text), **metadata},
        "_api_called": bool(parsed.get("_api_called")),
        "_used_fallback": bool(parsed.get("_used_fallback", not parsed.get("_api_called"))),
    }


def _enrich_single_call(text: str, source: str = "") -> dict:
    """One LLM call per chunk → summary + questions + context + metadata.

    Cost optimisation: 1 API call instead of 4. Returns the deterministic fallback shape
    when no API key is configured or the call fails.
    """
    if not OPENAI_API_KEY:
        return _normalize_result({"_used_fallback": True}, text, source)
    client = create_openai_client()
    if client is None:
        return _normalize_result({"_used_fallback": True}, text, source)
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL, temperature=0, max_tokens=500,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
            ])
        parsed = _parse_json_object(resp.choices[0].message.content)
        parsed["_api_called"] = True
        parsed["_used_fallback"] = False
        return _normalize_result(parsed, text, source)
    except Exception as exc:
        print(f"  ⚠️  Combined enrichment failed ({exc}); using fallback.")
        return _normalize_result({"_api_called": True, "_used_fallback": True}, text, source)


# ─── Cache ────────────────────────────────────────────────────────────────


def _cache_key(text: str, source: str) -> str:
    payload = f"{PROMPT_VERSION}|{OPENAI_MODEL}|{source}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    if not ENRICH_CACHE_ENABLED or not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    if not ENRICH_CACHE_ENABLED:
        return
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:
        pass


def _version_info(source: str) -> tuple[str | None, int | None]:
    m = re.match(r"(.+?)[_-]v?(20\d{2}|\d+)(?:\.[^.]+)?$", source, re.I)
    if not m:
        return None, None
    family = re.sub(r"[_-]+$", "", m.group(1)).lower()
    return family, int(m.group(2))


def _apply_version_signals(out: list[EnrichedChunk]) -> None:
    """Mark policy families so version-sensitive queries can prefer current documents."""
    families: dict[str, list[tuple[int, EnrichedChunk]]] = {}
    for item in out:
        fam, ver = _version_info(str(item.auto_metadata.get("source", "")))
        if fam is not None and ver is not None:
            families.setdefault(fam, []).append((ver, item))
    for fam, versions in families.items():
        if len(versions) < 2:
            continue
        current_ver = max(v for v, _ in versions)
        current_source = next(str(i.auto_metadata.get("source", "")) for v, i in versions if v == current_ver)
        for v, item in versions:
            current = v == current_ver
            item.auto_metadata.update({
                "policy_family": fam, "version": str(v), "is_current": current,
                "policy_status": "current" if current else "superseded",
                "superseded_by": None if current else current_source,
            })
            note = f"[Trạng thái phiên bản] {'Hiện hành' if current else 'Đã được thay thế bởi ' + current_source}."
            item.enriched_text = note + "\n\n" + item.enriched_text


# ─── Full enrichment pipeline ─────────────────────────────────────────────


def enrich_chunks(chunks: list[dict], methods: list[str] | None = None) -> list[EnrichedChunk]:
    """Enrich chunks and return `EnrichedChunk`s whose `original_text` is always preserved."""
    methods = methods or ["combined"]
    allowed = {"summary", "hyqa", "contextual", "metadata", "combined"}
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"Unknown enrichment methods: {sorted(unknown)}")
    use_combined = "combined" in methods

    valid = [c for c in chunks
             if isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip()]
    started = time.perf_counter()
    api_calls = cache_hits = fallbacks = 0
    combined: list[dict | None] = [None] * len(valid)

    if use_combined:
        cache = _load_cache()
        pending = []
        for i, c in enumerate(valid):
            text = c["text"].strip()
            source = str((c.get("metadata") or {}).get("source", ""))
            key = _cache_key(text, source)
            if key in cache:
                combined[i] = _normalize_result({**cache[key], "_api_called": False, "_used_fallback": False},
                                                text, source)
                cache_hits += 1
            else:
                pending.append((i, key, text, source))

        if pending:
            workers = max(1, min(ENRICH_MAX_WORKERS, len(pending)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_enrich_single_call, text, source): (i, key, text, source)
                           for i, key, text, source in pending}
                for future in as_completed(futures):
                    i, key, text, source = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        print(f"  ⚠️  Enrichment worker failed ({exc}); using fallback.")
                        result = _normalize_result({"_used_fallback": True}, text, source)
                    combined[i] = result
                    api_calls += int(bool(result.get("_api_called")))
                    fallbacks += int(bool(result.get("_used_fallback")))
                    if not result.get("_used_fallback"):
                        cache[key] = {k: result[k] for k in ("summary", "questions", "context", "metadata")}
            _save_cache(cache)

    out: list[EnrichedChunk] = []
    for i, c in enumerate(valid):
        text = c["text"].strip()
        base = dict(c.get("metadata") or {})
        source = str(base.get("source", ""))
        if use_combined:
            result = combined[i] or _normalize_result({"_used_fallback": True}, text, source)
            summary, hyqa = result["summary"], result["questions"]
            meta = result["metadata"]
            contextual = f"[Bối cảnh] {result['context']}\n\n{text}"
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            hyqa = generate_hypothesis_questions(text) if "hyqa" in methods else []
            meta = extract_metadata(text) if "metadata" in methods else {}
            contextual = contextual_prepend(text, source) if "contextual" in methods else text
        parts = []
        if summary:
            parts.append(f"[Tóm tắt] {summary}")
        if hyqa:
            parts.append("[Câu hỏi giả định] " + " | ".join(hyqa))
        parts.append(contextual)
        out.append(EnrichedChunk(text, "\n\n".join(parts), summary, hyqa, {**base, **meta},
                                 "combined" if use_combined else "+".join(methods)))

    _apply_version_signals(out)

    elapsed_ms = (time.perf_counter() - started) * 1000
    LAST_ENRICHMENT_STATS.clear()
    LAST_ENRICHMENT_STATS.update({
        "chunks": len(out),
        "mode": "combined" if use_combined else "individual",
        "api_calls": api_calls,
        "cache_hits": cache_hits,
        "fallbacks": fallbacks if use_combined else len(out),
        "workers": ENRICH_MAX_WORKERS,
        "total_ms": elapsed_ms,
        "avg_ms_per_chunk": elapsed_ms / max(1, len(out)),
    })
    return out
