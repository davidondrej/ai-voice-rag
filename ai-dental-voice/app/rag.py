import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import httpx
import faiss
import tiktoken

# Optional heavy imports guarded for file parsing
import fitz  # PyMuPDF
from docx import Document  # python-docx


# ---- Config ----
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBED_URL = "https://api.openai.com/v1/embeddings"

# Chunking defaults
CHUNK_TOKENS = int(os.environ.get("CHUNK_TOKENS", "400"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "40"))

# ----- Utilities -----
def _read_text_from_pdf(path: Path) -> str:
    with fitz.open(path) as doc:
        parts = []
        for page in doc:
            parts.append(page.get_text("text"))
    return "\n".join(parts).strip()


def _read_text_from_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs).strip()


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_text_from_pdf(path)
    if suffix == ".docx":
        return _read_text_from_docx(path)
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    # Unknown types not expected here
    return ""


def _get_encoder():
    # cl100k_base is appropriate for GPT-4/4o embeddings chunking
    return tiktoken.get_encoding("cl100k_base")


def chunk_by_tokens(text: str, chunk_tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    enc = _get_encoder()
    toks = enc.encode(text)
    if not toks:
        return []
    chunks = []
    start = 0
    while start < len(toks):
        end = min(start + chunk_tokens, len(toks))
        piece = enc.decode(toks[start:end])
        chunks.append(piece)
        if end == len(toks):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


async def embed_texts(texts: List[str], api_key: str, model: str = EMBED_MODEL, batch: int = 64) -> np.ndarray:
    """
    Returns float32 numpy array of shape (n, d)
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    out: List[List[float]] = []
    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(0, len(texts), batch):
            payload = {
                "model": model,
                "input": texts[i:i + batch],
            }
            r = await client.post(EMBED_URL, headers=headers, json=payload)
            if r.status_code != 200:
                raise RuntimeError(f"Embeddings error: {r.status_code} {r.text}")
            data = r.json()["data"]
            for item in data:
                out.append(item["embedding"])
    arr = np.array(out, dtype="float32")
    return arr


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vecs / norms


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class RagIndex:
    """
    Flat cosine-similarity index using FAISS IndexFlatIP with L2-normalized vectors.
    Metadata is stored in a parallel JSON file.
    """
    def __init__(self, index_path: Path, meta_path: Path):
        self.index_path = index_path
        self.meta_path = meta_path
        self.index: Optional[faiss.IndexFlatIP] = None
        self.dim: Optional[int] = None
        self.meta: Dict = {"dim": None, "items": []}

    def load(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        meta = _load_json(self.meta_path, {"dim": None, "items": []})
        self.meta = meta
        self.dim = meta.get("dim")

    def _ensure_index(self, dim: int):
        if self.index is not None:
            return
        self.index = faiss.IndexFlatIP(dim)
        self.dim = dim
        self.meta["dim"] = dim

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        _save_json(self.meta_path, self.meta)

    def add(self, vecs: np.ndarray, metadatas: List[Dict]):
        if vecs.size == 0:
            return
        vecs = _l2_normalize(vecs.astype("float32"))
        self._ensure_index(vecs.shape[1])
        if self.index is None:
            raise RuntimeError("FAISS index not initialized")
        # Append vectors
        self.index.add(vecs)
        # Append metadata in the same order
        self.meta["items"].extend(metadatas)

    def count(self) -> int:
        return 0 if self.index is None else self.index.ntotal

    def search(self, query_vec: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        if self.index is None or self.count() == 0:
            return []
        q = _l2_normalize(query_vec.astype("float32"))
        D, I = self.index.search(q, k)  # inner-product scores on normalized vectors == cosine similarity
        idxs = I[0].tolist()
        scores = D[0].tolist()
        out = []
        for i, s in zip(idxs, scores):
            if i == -1:
                continue
            out.append((i, float(s)))
        return out


async def process_and_index_document(
    rag: RagIndex,
    file_path: Path,
    folder: str,
    api_key: str,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap: int = CHUNK_OVERLAP,
) -> Dict:
    text = extract_text(file_path)
    if not text.strip():
        return {"added": 0, "chunks": 0, "note": "no text extracted"}

    chunks = chunk_by_tokens(text, chunk_tokens, overlap)
    if not chunks:
        return {"added": 0, "chunks": 0, "note": "chunking produced 0"}

    vecs = await embed_texts(chunks, api_key=api_key, model=EMBED_MODEL)
    metas = []
    fname = file_path.name
    for i, c in enumerate(chunks):
        metas.append({
            "folder": folder,
            "filename": fname,
            "chunk_index": i,
            "tokens": None,  # optional
            "text": c,
            "source": f"/files/docs/{folder}/{fname}",
        })

    rag.add(vecs, metas)
    rag.save()
    return {"added": len(chunks), "chunks": len(chunks)}


