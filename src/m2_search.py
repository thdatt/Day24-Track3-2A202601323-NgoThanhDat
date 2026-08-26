from __future__ import annotations

"""M2 — Vietnamese BM25 + dense retrieval + Reciprocal Rank Fusion."""

import hashlib
import math
import re
import uuid
from collections import Counter
from dataclasses import dataclass

from config import (BM25_TOP_K, COLLECTION_NAME, DENSE_TOP_K, EMBEDDING_DIM, EMBEDDING_MODEL,
                    HYBRID_TOP_K, QDRANT_HOST, QDRANT_PORT)

@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str


def segment_vietnamese(text: str) -> str:
    normalized = " ".join(text.lower().split())
    if not normalized:
        return ""
    try:
        from underthesea import word_tokenize
        return " ".join(word_tokenize(normalized, format="text").replace("_", " ").split())
    except Exception:
        return normalized


class _SimpleBM25:
    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus; self.k1 = k1; self.b = b
        self.lengths = [len(x) for x in corpus]
        self.avgdl = sum(self.lengths) / max(1, len(self.lengths))
        self.tf = [Counter(x) for x in corpus]
        df = Counter(term for doc in corpus for term in set(doc))
        n = max(1, len(corpus))
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
    def get_scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, dl in zip(self.tf, self.lengths):
            score = 0.0
            norm = 1 - self.b + self.b * dl / max(self.avgdl, 1)
            for t in query:
                freq = tf.get(t, 0)
                if freq:
                    score += self.idf.get(t, 0.0) * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)
            out.append(score)
        return out


class BM25Search:
    def __init__(self):
        self.documents: list[dict] = []
        self.bm25 = None
    def index(self, chunks: list[dict]) -> None:
        self.documents = [{"text": c["text"].strip(), "metadata": dict(c.get("metadata", {}))}
                          for c in chunks if isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip()]
        corpus = [segment_vietnamese(d["text"]).split() for d in self.documents]
        if not corpus:
            self.bm25 = None; return
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(corpus)
        except Exception:
            self.bm25 = _SimpleBM25(corpus)
    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        if self.bm25 is None or top_k <= 0:
            return []
        q = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(q) if q else []
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        out = []
        for i in order:
            score = float(scores[i])
            if score <= 0:
                continue
            d = self.documents[i]
            out.append(SearchResult(d["text"], score, dict(d["metadata"]), "bm25"))
            if len(out) >= top_k:
                break
        return out


def _hash_vector(text: str, dim: int = 256) -> list[float]:
    v = [0.0] * dim
    for token in re.findall(r"\w+", segment_vietnamese(text), re.UNICODE):
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class DenseSearch:
    """Qdrant-backed dense search with an automatic in-memory fallback."""
    def __init__(self):
        self._encoder = None
        self._memory: dict[str, list[tuple[dict, list[float]]]] = {}
        self.client = None
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=1.5)
            self.client.get_collections()
        except Exception:
            self.client = None
    def _encode(self, texts):
        try:
            if self._encoder is None:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            return self._encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        except Exception:
            if isinstance(texts, str):
                return _hash_vector(texts)
            return [_hash_vector(t) for t in texts]
    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        valid = [{"text": c["text"].strip(), "metadata": dict(c.get("metadata", {}))}
                 for c in chunks if isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip()]
        if not valid:
            raise ValueError("Cannot index an empty chunk list")
        vectors = self._encode([c["text"] for c in valid])
        if self.client is None:
            self._memory[collection] = list(zip(valid, [list(v) for v in vectors])); return
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams
            dim = len(vectors[0])
            if self.client.collection_exists(collection):
                self.client.delete_collection(collection)
            self.client.create_collection(collection_name=collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
            points = []
            for c, v in zip(valid, vectors):
                ident = str(uuid.uuid5(uuid.NAMESPACE_URL, c["text"] + repr(c["metadata"])))
                points.append(PointStruct(id=ident, vector=list(v), payload={**c["metadata"], "text": c["text"]}))
            self.client.upsert(collection_name=collection, points=points, wait=True)
        except Exception:
            self.client = None
            self._memory[collection] = list(zip(valid, [list(v) for v in vectors]))
    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        if top_k <= 0 or not query.strip():
            return []
        qv = list(self._encode(query))
        if self.client is None:
            pairs = self._memory.get(collection, [])
            scored = sorted(((d, _cos(qv, v)) for d, v in pairs), key=lambda x: x[1], reverse=True)
            return [SearchResult(d["text"], float(s), dict(d["metadata"]), "dense") for d, s in scored[:top_k]]
        try:
            resp = self.client.query_points(collection_name=collection, query=qv, limit=top_k, with_payload=True)
            out = []
            for p in resp.points:
                payload = dict(p.payload or {}); text = str(payload.pop("text", "")).strip()
                if text:
                    out.append(SearchResult(text, float(p.score), payload, "dense"))
            return out
        except Exception:
            return []


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []
    scores: dict[tuple, float] = {}
    best: dict[tuple, SearchResult] = {}
    for results in results_list:
        for rank, item in enumerate(results, start=1):
            key = (item.text, str(item.metadata.get("source", "")), str(item.metadata.get("parent_id", "")))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            best.setdefault(key, item)
    order = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [SearchResult(best[key].text, scores[key], dict(best[key].metadata), "hybrid") for key in order]


class HybridSearch:
    def __init__(self):
        self.bm25 = BM25Search(); self.dense = DenseSearch(); self.collection = COLLECTION_NAME
    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME):
        self.collection = collection; self.bm25.index(chunks); self.dense.index(chunks, collection)
    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        return reciprocal_rank_fusion([
            self.bm25.search(query, BM25_TOP_K),
            self.dense.search(query, DENSE_TOP_K, self.collection),
        ], top_k=top_k)
