from __future__ import annotations

import json
import math
import re
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize(text: str) -> List[str]:
    lowered = text.lower()
    tokens: List[str] = []
    tokens.extend(re.findall(r"[a-z0-9_]+", lowered))

    cjk = re.findall(r"[\u4e00-\u9fff]", lowered)
    tokens.extend(cjk)
    if len(cjk) >= 2:
        tokens.extend("".join(cjk[i : i + 2]) for i in range(len(cjk) - 1))
    return tokens


def _chunk_text(text: str, size: int = 600, overlap: int = 80) -> List[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []

    chunks: List[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(clean), step):
        piece = clean[start : start + size].strip()
        if piece:
            chunks.append(piece)
        if start + size >= len(clean):
            break
    return chunks


class RAGStore:
    def __init__(self, file_path: str, max_documents: int = 2000, max_chunks: int = 30000) -> None:
        self.file_path = Path(file_path)
        self.max_documents = max(1, int(max_documents))
        self.max_chunks = max(1, int(max_chunks))
        self._lock = threading.Lock()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text(json.dumps({"documents": [], "chunks": []}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _load(self) -> Dict[str, Any]:
        raw = self.file_path.read_text(encoding="utf-8").strip()
        if not raw:
            return {"documents": [], "chunks": []}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            backup = self.file_path.with_suffix(self.file_path.suffix + f".corrupt.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
            backup.write_text(raw, encoding="utf-8")
            return {"documents": [], "chunks": []}
        if not isinstance(data, dict):
            return {"documents": [], "chunks": []}
        documents = data.get("documents")
        chunks = data.get("chunks")
        if not isinstance(documents, list) or not isinstance(chunks, list):
            return {"documents": [], "chunks": []}
        return {"documents": documents, "chunks": chunks}

    def _save(self, data: Dict[str, Any]) -> None:
        self.file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def add_document(
        self,
        *,
        file_name: str,
        source_type: str,
        import_type: str,
        segments: Iterable[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        chunks_payload: List[Dict[str, Any]] = []
        for seg in segments:
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            seg_meta = seg.get("metadata") if isinstance(seg.get("metadata"), dict) else {}
            for idx, piece in enumerate(_chunk_text(text)):
                tf = Counter(_tokenize(piece))
                chunks_payload.append(
                    {
                        "id": str(uuid.uuid4()),
                        "text": piece,
                        "metadata": {**seg_meta, "segment_index": idx},
                        "tf": dict(tf),
                        "len": sum(tf.values()) or 1,
                    }
                )

        if not chunks_payload:
            raise ValueError("解析后没有有效文本，无法入库")

        doc = {
            "id": str(uuid.uuid4()),
            "file_name": file_name,
            "source_type": source_type,
            "import_type": import_type,
            "metadata": metadata or {},
            "created_at": _now_iso(),
            "chunk_count": len(chunks_payload),
        }

        with self._lock:
            data = self._load()
            for chunk in chunks_payload:
                chunk["doc_id"] = doc["id"]
            data["documents"].append(doc)
            data["chunks"].extend(chunks_payload)

            if len(data["documents"]) > self.max_documents:
                keep_docs = data["documents"][-self.max_documents :]
                keep_doc_ids = {x.get("id") for x in keep_docs}
                data["documents"] = keep_docs
                data["chunks"] = [x for x in data["chunks"] if x.get("doc_id") in keep_doc_ids]
            if len(data["chunks"]) > self.max_chunks:
                data["chunks"] = data["chunks"][-self.max_chunks :]
                keep_doc_ids = {x.get("doc_id") for x in data["chunks"]}
                data["documents"] = [x for x in data["documents"] if x.get("id") in keep_doc_ids]

            # refresh chunk_count after pruning
            counts: Dict[str, int] = {}
            for c in data["chunks"]:
                did = str(c.get("doc_id", ""))
                counts[did] = counts.get(did, 0) + 1
            for d in data["documents"]:
                d["chunk_count"] = counts.get(str(d.get("id", "")), 0)

            self._save(data)

        return doc

    def list_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._load()
        docs = sorted(data["documents"], key=lambda x: str(x.get("created_at", "")), reverse=True)
        return docs[: max(1, min(limit, 500))]

    def search(self, query: str, top_k: int = 5, source_type: Optional[str] = None) -> List[Dict[str, Any]]:
        query = str(query or "").strip()
        if not query:
            return []

        with self._lock:
            data = self._load()
        chunks = data["chunks"]
        docs_by_id = {str(d.get("id")): d for d in data["documents"]}
        if source_type:
            chunks = [x for x in chunks if (docs_by_id.get(str(x.get("doc_id"))) or {}).get("source_type") == source_type]
        if not chunks:
            return []

        q_tf = Counter(_tokenize(query))
        if not q_tf:
            return []

        n = len(chunks)
        df: Dict[str, int] = {}
        for chunk in chunks:
            tf = chunk.get("tf") if isinstance(chunk.get("tf"), dict) else {}
            for token in tf.keys():
                df[token] = df.get(token, 0) + 1

        q_weights: Dict[str, float] = {}
        for token, freq in q_tf.items():
            idf = math.log((n + 1) / (df.get(token, 0) + 1)) + 1.0
            q_weights[token] = float(freq) * idf
        q_norm = math.sqrt(sum(v * v for v in q_weights.values())) or 1.0

        scored: List[Dict[str, Any]] = []
        for chunk in chunks:
            tf = chunk.get("tf") if isinstance(chunk.get("tf"), dict) else {}
            if not tf:
                continue

            dot = 0.0
            c_norm_sq = 0.0
            for token, freq in tf.items():
                idf = math.log((n + 1) / (df.get(token, 0) + 1)) + 1.0
                weight = float(freq) * idf
                c_norm_sq += weight * weight
                if token in q_weights:
                    dot += q_weights[token] * weight
            c_norm = math.sqrt(c_norm_sq) or 1.0
            score = dot / (q_norm * c_norm)
            if score <= 0:
                continue

            doc = docs_by_id.get(str(chunk.get("doc_id"))) or {}
            scored.append(
                {
                    "chunk_id": chunk.get("id"),
                    "doc_id": chunk.get("doc_id"),
                    "doc_name": doc.get("file_name"),
                    "source_type": doc.get("source_type"),
                    "import_type": doc.get("import_type"),
                    "text": chunk.get("text"),
                    "metadata": chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {},
                    "score": round(score, 6),
                }
            )

        scored.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return scored[: max(1, min(top_k, 20))]

