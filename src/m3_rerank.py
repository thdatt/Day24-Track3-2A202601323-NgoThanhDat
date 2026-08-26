from __future__ import annotations

"""M3 — cross-encoder or FlashRank reranking with lexical fallback."""
import re
import time
from dataclasses import dataclass
from statistics import mean

from config import FLASHRANK_CACHE_DIR, FLASHRANK_MODEL, RERANK_BACKEND, RERANK_BATCH_SIZE, RERANK_DEVICE, RERANK_MODEL, RERANK_TOP_K

@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


def _lexical_score(query: str, text: str) -> float:
    # Do not let generated HyQA boilerplate dominate the offline fallback.
    clean_text = re.sub(r"\[Câu hỏi giả định\][^\n]*", " ", text, flags=re.IGNORECASE).lower()
    query_l = query.lower()
    stop = {
        "nhân", "viên", "được", "bao", "nhiêu", "sau", "mới", "bị", "có",
        "một", "theo", "và", "là", "thì", "của", "trong", "phải", "đi",
        "khi", "này", "đó", "cần", "không", "để", "mỗi", "cho", "với",
    }
    q_tokens = re.findall(r"\w+", query_l, re.UNICODE)
    d_tokens = set(re.findall(r"\w+", clean_text, re.UNICODE))
    q_terms = [t for t in q_tokens if t not in stop and len(t) > 1] or q_tokens
    if not q_terms:
        return 0.0

    weights = {t: 1.0 + min(len(t), 8) / 8.0 for t in set(q_terms)}
    denom = sum(weights.values()) or 1.0
    overlap = sum(w for t, w in weights.items() if t in d_tokens) / denom

    phrase_bonus = 0.0
    meaningful = [t for t in q_tokens if t not in stop]
    for first, second in zip(meaningful, meaningful[1:]):
        if len(first) > 1 and len(second) > 1 and f"{first} {second}" in clean_text:
            phrase_bonus += 0.12
    phrase_bonus = min(phrase_bonus, 0.36)

    numbers_q = set(re.findall(r"\d+[\d.,]*", query_l))
    numbers_d = set(re.findall(r"\d+[\d.,]*", clean_text))
    number_bonus = 0.30 * (len(numbers_q & numbers_d) / len(numbers_q)) if numbers_q else 0.0
    return overlap + phrase_bonus + number_bonus


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANK_MODEL, device: str | None = None):
        self.model_name = model_name; self.device = device or RERANK_DEVICE; self.batch_size = RERANK_BATCH_SIZE
        self._model = None; self.last_rerank_ms = 0.0; self.last_used_fallback = False
    @staticmethod
    def _model_text(doc: dict) -> str:
        meta = doc.get("metadata", {}) or {}
        prefix = " ".join(str(meta.get(k, "")) for k in ("source", "section", "topic") if meta.get(k))
        return f"{prefix}\n{doc['text']}".strip()
    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model
    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        valid = [d for d in documents if isinstance(d, dict) and isinstance(d.get("text"), str) and d["text"].strip()]
        if not query.strip() or not valid or top_k <= 0:
            return []
        started = time.perf_counter()
        try:
            model = self._load_model()
            raw = model.predict([(query, self._model_text(d)) for d in valid], batch_size=self.batch_size, show_progress_bar=False)
            scores = [float(x) for x in raw]
            self.last_used_fallback = False
        except Exception:
            scores = [_lexical_score(query, d["text"]) for d in valid]
            self.last_used_fallback = True
        ranked = sorted(zip(valid, scores), key=lambda x: x[1], reverse=True)[:top_k]
        self.last_rerank_ms = (time.perf_counter() - started) * 1000
        return [RerankResult(d["text"], float(d.get("score", 0.0) or 0.0), float(s), dict(d.get("metadata", {})), i)
                for i, (d, s) in enumerate(ranked, start=1)]


class FlashrankReranker(CrossEncoderReranker):
    def __init__(self, model_name: str = FLASHRANK_MODEL, cache_dir: str = FLASHRANK_CACHE_DIR, max_length: int = 512):
        super().__init__(model_name=model_name, device="cpu")
        self.cache_dir = cache_dir; self.max_length = max_length
    def _load_model(self):
        if self._model is None:
            from flashrank import Ranker
            self._model = Ranker(model_name=self.model_name, cache_dir=self.cache_dir, max_length=self.max_length, log_level="WARNING")
        return self._model
    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        valid = [d for d in documents if isinstance(d, dict) and isinstance(d.get("text"), str) and d["text"].strip()]
        if not query.strip() or not valid or top_k <= 0:
            return []
        started = time.perf_counter()
        try:
            from flashrank import RerankRequest
            passages = [{"id": str(i), "text": self._model_text(d)} for i, d in enumerate(valid)]
            raw = self._load_model().rerank(RerankRequest(query=query, passages=passages))
            ranked = []
            for item in raw[:top_k]:
                idx = int(item.get("id", 0)); d = valid[idx]
                ranked.append((d, float(item.get("score", 0.0))))
            self.last_used_fallback = False
        except Exception:
            ranked = sorted(((d, _lexical_score(query, d["text"])) for d in valid), key=lambda x: x[1], reverse=True)[:top_k]
            self.last_used_fallback = True
        self.last_rerank_ms = (time.perf_counter() - started) * 1000
        return [RerankResult(d["text"], float(d.get("score", 0.0) or 0.0), s, dict(d.get("metadata", {})), i)
                for i, (d, s) in enumerate(ranked, start=1)]


def create_reranker(backend: str | None = None):
    backend = (backend or RERANK_BACKEND).lower()
    if backend in {"cross_encoder", "cross-encoder", "bge"}:
        return CrossEncoderReranker()
    if backend in {"flashrank", "flash_rank"}:
        return FlashrankReranker()
    raise ValueError(f"Unknown reranker backend: {backend}")


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 3) -> dict:
    timings = []
    for _ in range(max(1, n_runs)):
        t0 = time.perf_counter(); reranker.rerank(query, documents); timings.append((time.perf_counter() - t0) * 1000)
    return {"avg_ms": mean(timings), "min_ms": min(timings), "max_ms": max(timings), "runs": len(timings)}
