from __future__ import annotations

"""M1 — advanced chunking: semantic, hierarchical and structure-aware."""

import glob
import hashlib
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable

from config import DATA_DIR, HIERARCHICAL_CHILD_SIZE, HIERARCHICAL_PARENT_SIZE, SEMANTIC_THRESHOLD

@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    reader = PdfReader(path)
    return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    docs: list[dict] = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        try:
            text = _extract_pdf_text(fp)
        except Exception as exc:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: {exc}")
            continue
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan/không có text layer.")
    return docs


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    metadata = metadata or {}
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[Chunk] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if current and len(candidate) > chunk_size:
            out.append(Chunk(current.strip(), {**metadata, "chunk_index": len(out), "strategy": "basic"}))
            current = para
        else:
            current = candidate
    if current:
        out.append(Chunk(current.strip(), {**metadata, "chunk_index": len(out), "strategy": "basic"}))
    return out


@lru_cache(maxsize=1)
def _semantic_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _split_sentences(text: str) -> list[str]:
    units = [x.strip() for x in re.split(r"(?<=[!?])\s+|(?<!\d)(?<=\.)\s+|\n{2,}", text.strip()) if x.strip()]
    merged: list[str] = []
    i = 0
    while i < len(units):
        if re.fullmatch(r"#{1,6}\s+[^\n]+", units[i]) and i + 1 < len(units):
            merged.append(units[i] + "\n" + units[i + 1]); i += 2
        else:
            merged.append(units[i]); i += 1
    compact: list[str] = []
    for u in merged:
        if len(u) < 30 and compact:
            compact[-1] += " " + u
        else:
            compact.append(u)
    return compact


def _token_similarity(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower(), re.UNICODE))
    tb = set(re.findall(r"\w+", b.lower(), re.UNICODE))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD, metadata: dict | None = None) -> list[Chunk]:
    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(sentences[0], {**metadata, "strategy": "semantic", "chunk_index": 0, "sentence_count": 1})]

    similarities: list[float] = []
    try:
        emb = _semantic_model().encode(sentences, normalize_embeddings=True, show_progress_bar=False)
        similarities = [float(emb[i - 1] @ emb[i]) for i in range(1, len(sentences))]
    except Exception:
        # Offline fallback uses lexical overlap and a gentler effective threshold.
        similarities = [_token_similarity(sentences[i - 1], sentences[i]) for i in range(1, len(sentences))]
        threshold = min(threshold, 0.12)

    groups = [[sentences[0]]]
    for sentence, similarity in zip(sentences[1:], similarities):
        if similarity < threshold:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)
    return [
        Chunk(" ".join(group), {**metadata, "strategy": "semantic", "chunk_index": i,
                                  "sentence_count": len(group), "similarity_threshold": threshold})
        for i, group in enumerate(groups)
    ]


def _split_text_by_size(text: str, max_size: int) -> list[str]:
    if max_size <= 0:
        raise ValueError("max_size must be > 0")
    text = text.strip()
    if not text:
        return []
    out: list[str] = []
    rest = text
    while len(rest) > max_size:
        target = max_size
        cut = -1
        for marker in ("\n\n", "\n", ". ", "; ", ", ", " "):
            pos = rest.rfind(marker, 0, target + 1)
            if pos >= target // 2:
                cut = pos + (0 if marker.isspace() else 1)
                break
        if cut <= 0:
            cut = target
        out.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        out.append(rest)
    return [x for x in out if x]


def _parent_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{text}".encode()).hexdigest()[:12]
    safe = re.sub(r"[^\w.-]+", "_", source or "document")
    return f"{safe}::parent::{index}::{digest}"


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    if parent_size <= 0 or child_size <= 0:
        raise ValueError("chunk sizes must be positive")
    if child_size >= parent_size:
        raise ValueError("child_size must be smaller than parent_size")
    metadata = metadata or {}
    parents: list[Chunk] = []
    children: list[Chunk] = []
    source = str(metadata.get("source", "document"))
    for pidx, ptext in enumerate(_split_text_by_size(text, parent_size)):
        pid = _parent_id(source, pidx, ptext)
        parent = Chunk(ptext, {**metadata, "strategy": "hierarchical", "chunk_type": "parent",
                               "chunk_index": pidx, "parent_id": pid}, pid)
        parents.append(parent)
        for cidx, ctext in enumerate(_split_text_by_size(ptext, child_size)):
            children.append(Chunk(ctext, {**metadata, "strategy": "hierarchical", "chunk_type": "child",
                                          "chunk_index": len(children), "child_index": cidx, "parent_id": pid}, pid))
    return parents, children


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    metadata = metadata or {}
    text = text.strip()
    if not text:
        return []
    pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    out: list[Chunk] = []
    path: list[str] = []

    def add(part: str, title: str, level: int, p: list[str]):
        part = part.strip()
        if part:
            out.append(Chunk(part, {**metadata, "strategy": "structure", "chunk_index": len(out),
                                    "section": title, "section_path": " > ".join(p) if p else title,
                                    "header_level": level}))

    if not matches:
        add(text, "preamble", 0, [])
        return out
    add(text[:matches[0].start()], "preamble", 0, [])
    for i, m in enumerate(matches):
        level = len(m.group(1)); title = m.group(2).strip()
        path = path[:level - 1] + [title]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        add(text[m.start():end], title, level, path.copy())
    return out


def compare_strategies(documents: list[dict]) -> dict:
    text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}
    basic = chunk_basic(text, metadata=meta)
    semantic = chunk_semantic(text, metadata=meta)
    parents, children = chunk_hierarchical(text, metadata=meta)
    structure = chunk_structure_aware(text, metadata=meta)

    def stats(items: Iterable[Chunk]) -> dict:
        lengths = [len(c.text) for c in items]
        return {"count": len(lengths), "avg_len": round(sum(lengths) / len(lengths)) if lengths else 0,
                "min_len": min(lengths) if lengths else 0, "max_len": max(lengths) if lengths else 0}

    return {"basic": stats(basic), "semantic": stats(semantic),
            "hierarchical": {**stats(children), "parents": len(parents)}, "structure": stats(structure)}
