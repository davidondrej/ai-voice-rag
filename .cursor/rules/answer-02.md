Use WebRTC. Keep everything in one FastAPI app. Add minimal RAG with FAISS and on‑upload ingestion.

**Updated file tree**

```
ai-dental-voice/
  app/
    main.py
    rag.py
  static/
    index.html
    app.js
  data/
    .gitkeep
    docs/
      (folders created on demand)
    index.faiss        # created after first ingestion
    index_meta.json    # created after first ingestion
  .env.example
  .gitignore
  Dockerfile
  requirements.txt
```

**app/rag.py**

```python
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
```

**app/main.py**

```python
import os
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .rag import RagIndex, process_and_index_document, embed_texts

# --- Config ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
REALTIME_MODEL = os.environ.get("REALTIME_MODEL", "gpt-4o-realtime-preview-2025-06-03")
REALTIME_VOICE = os.environ.get("REALTIME_VOICE", "verse")

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
DOCS_ROOT = DATA_DIR / "docs"
INDEX_PATH = DATA_DIR / "index.faiss"
META_PATH = DATA_DIR / "index_meta.json"

DOCS_ROOT.mkdir(parents=True, exist_ok=True)

# --- App ---
app = FastAPI(title="AI Dental Voice - Monolith")

# CORS (dev-friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static mounts
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")
app.mount("/files", StaticFiles(directory=DATA_DIR, html=False), name="files")

# RAG index
rag = RagIndex(INDEX_PATH, META_PATH)
rag.load()

# Root -> UI
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = STATIC_DIR / "index.html"
    return index_path.read_text(encoding="utf-8")


# 1) Ephemeral Realtime session
@app.get("/session")
async def create_realtime_session():
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    url = "https://api.openai.com/v1/realtime/sessions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": REALTIME_MODEL,
        "voice": REALTIME_VOICE,
    }

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(url, headers=headers, json=body)
    return JSONResponse(status_code=resp.status_code, content=resp.json())


# 2) Upload documents with optional folder (default=default). Index immediately.
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt"}

@app.post("/upload")
async def upload(
    files: List[UploadFile] = File(...),
    folder: str = Form("default"),
):
    if not folder:
        folder = "default"
    # Sanitize folder to a simple name
    folder = "".join(ch for ch in folder if ch.isalnum() or ch in ("-", "_")).strip() or "default"

    folder_dir = DOCS_ROOT / folder
    folder_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        name = Path(f.filename).name
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

        dest = folder_dir / name
        if dest.exists():
            stem = dest.stem
            suffix_txt = dest.suffix
            i = 1
            while True:
                candidate = folder_dir / f"{stem} ({i}){suffix_txt}"
                if not candidate.exists():
                    dest = candidate
                    break
                i += 1

        with dest.open("wb") as out:
            while True:
                chunk = await f.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        # Ingest + index immediately
        if not OPENAI_API_KEY:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
        ingest_info = await process_and_index_document(
            rag=rag,
            file_path=dest,
            folder=folder,
            api_key=OPENAI_API_KEY,
        )

        saved.append({
            "name": dest.name,
            "size": dest.stat().st_size,
            "folder": folder,
            "url": f"/files/docs/{folder}/{dest.name}",
            "ingest": ingest_info,
        })

    return {"saved": saved}


# 3) List uploaded documents; optional ?folder=
@app.get("/list")
async def list_docs(folder: Optional[str] = Query(default=None)):
    items = []
    folders = []
    if folder:
        base = DOCS_ROOT / folder
        if base.exists() and base.is_dir():
            folders = [folder]
            for p in sorted(base.iterdir()):
                if p.is_file():
                    items.append({
                        "folder": folder,
                        "name": p.name,
                        "size": p.stat().st_size,
                        "url": f"/files/docs/{folder}/{p.name}",
                    })
    else:
        for d in sorted(DOCS_ROOT.iterdir()):
            if d.is_dir():
                folders.append(d.name)
                for p in sorted(d.iterdir()):
                    if p.is_file():
                        items.append({
                            "folder": d.name,
                            "name": p.name,
                            "size": p.stat().st_size,
                            "url": f"/files/docs/{d.name}/{p.name}",
                        })
    return {"folders": folders, "docs": items}


# 4) RAG search: ?q=...&k=5
@app.get("/rag/search")
async def rag_search(q: str = Query(..., min_length=1), k: int = Query(5, ge=1, le=20)):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    if rag.count() == 0:
        return {"query": q, "results": [], "note": "index empty"}

    # Embed query
    q_vec = await embed_texts([q], api_key=OPENAI_API_KEY)
    hits = rag.search(q_vec, k=k)

    results = []
    for idx, score in hits:
        if idx < 0 or idx >= len(rag.meta["items"]):
            continue
        m = rag.meta["items"][idx]
        results.append({
            "text": m.get("text", ""),
            "score": score,
            "source": m.get("source"),
            "folder": m.get("folder"),
            "filename": m.get("filename"),
            "chunk_index": m.get("chunk_index"),
        })
    return {"query": q, "results": results}
```

**static/index.html**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>AI Dental Voice Prototype</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 text-slate-900">
  <div class="max-w-3xl mx-auto p-6 space-y-8">
    <header class="space-y-2">
      <h1 class="text-2xl font-semibold">AI Dental Voice Prototype</h1>
      <p class="text-sm text-slate-600">WebRTC voice agent + uploads + minimal RAG.</p>
    </header>

    <section class="p-4 bg-white rounded-lg border space-y-4">
      <h2 class="font-medium">Voice Agent</h2>
      <div class="flex items-center gap-2">
        <button id="btnStart" class="px-4 py-2 rounded bg-black text-white">Start</button>
        <button id="btnStop" class="px-4 py-2 rounded border">Stop</button>
        <span id="status" class="text-sm text-slate-600">idle</span>
      </div>
      <audio id="remoteAudio" autoplay></audio>
      <div class="text-sm text-slate-600">
        <div>Mic: <span id="micState">off</span></div>
        <div>Connection: <span id="connState">disconnected</span></div>
      </div>
      <pre id="log" class="text-xs bg-slate-100 p-2 rounded overflow-auto max-h-40"></pre>
    </section>

    <section class="p-4 bg-white rounded-lg border space-y-4">
      <h2 class="font-medium">Upload Documents</h2>
      <label class="text-sm text-slate-700">Folder (optional)</label>
      <input id="folderInput" type="text" placeholder="default" class="block border rounded px-2 py-1 w-48" />
      <input id="inputFiles" type="file" accept=".pdf,.docx,.txt" multiple class="block" />
      <button id="btnUpload" class="px-4 py-2 rounded bg-black text-white">Upload & Index</button>
      <div id="uploadMsg" class="text-sm text-slate-600"></div>
    </section>

    <section class="p-4 bg-white rounded-lg border space-y-4">
      <h2 class="font-medium">Documents</h2>
      <div id="docsList" class="space-y-2 text-sm"></div>
    </section>

    <section class="p-4 bg-white rounded-lg border space-y-4">
      <h2 class="font-medium">RAG Search</h2>
      <div class="flex gap-2">
        <input id="searchInput" type="text" placeholder="Ask a question..." class="flex-1 border rounded px-2 py-1" />
        <button id="btnSearch" class="px-4 py-2 rounded border">Search</button>
      </div>
      <div id="searchOut" class="text-sm text-slate-700 space-y-3"></div>
    </section>

  </div>
  <script src="/static/app.js"></script>
</body>
</html>
```

**static/app.js**

```javascript
let pc = null;
let dc = null;
let localStream = null;

const logEl = document.getElementById("log");
const statusEl = document.getElementById("status");
const micEl = document.getElementById("micState");
const connEl = document.getElementById("connState");
const remoteAudio = document.getElementById("remoteAudio");

function log(s) {
  logEl.textContent += s + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

async function startVoice() {
  statusEl.textContent = "starting...";
  connEl.textContent = "connecting";
  try {
    const r = await fetch("/session");
    if (!r.ok) throw new Error("failed /session: " + (await r.text()));
    const data = await r.json();
    const EPHEMERAL_KEY = data?.client_secret?.value;
    if (!EPHEMERAL_KEY) throw new Error("no ephemeral key");

    pc = new RTCPeerConnection();
    pc.onconnectionstatechange = () => connEl.textContent = pc.connectionState;
    pc.ontrack = (e) => remoteAudio.srcObject = e.streams[0];

    localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    micEl.textContent = "on";
    pc.addTrack(localStream.getTracks()[0]);

    dc = pc.createDataChannel("oai-events");
    dc.onmessage = (e) => log("[event] " + e.data);

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const baseUrl = "https://api.openai.com/v1/realtime";
    const model = "gpt-4o-realtime-preview-2025-06-03";
    const sdpResp = await fetch(`${baseUrl}?model=${encodeURIComponent(model)}`, {
      method: "POST",
      body: offer.sdp,
      headers: { Authorization: `Bearer ${EPHEMERAL_KEY}`, "Content-Type": "application/sdp" },
    });
    if (!sdpResp.ok) throw new Error("SDP exchange failed: " + (await sdpResp.text()));

    const answer = { type: "answer", sdp: await sdpResp.text() };
    await pc.setRemoteDescription(answer);

    statusEl.textContent = "connected";
    log("voice ready");
  } catch (err) {
    statusEl.textContent = "error";
    log("ERROR: " + (err?.message || String(err)));
    await stopVoice();
  }
}

async function stopVoice() {
  try {
    if (dc) { dc.close(); dc = null; }
    if (pc) { pc.close(); pc = null; }
    if (localStream) { localStream.getTracks().forEach(t => t.stop()); localStream = null; }
  } finally {
    micEl.textContent = "off";
    connEl.textContent = "disconnected";
    statusEl.textContent = "idle";
  }
}

async function refreshDocs() {
  const r = await fetch("/list");
  const data = await r.json();
  const list = document.getElementById("docsList");
  list.innerHTML = "";
  if (!data.docs || data.docs.length === 0) {
    list.textContent = "No documents uploaded yet.";
    return;
  }
  for (const d of data.docs) {
    const a = document.createElement("a");
    a.href = d.url;
    a.textContent = `${d.folder}/${d.name} (${d.size} bytes)`;
    a.className = "text-blue-600 underline";
    a.target = "_blank";
    const div = document.createElement("div");
    div.appendChild(a);
    list.appendChild(div);
  }
}

async function uploadFiles() {
  const files = document.getElementById("inputFiles").files;
  const folder = (document.getElementById("folderInput").value || "default").trim();
  if (!files || files.length === 0) return;
  const form = new FormData();
  for (const f of files) form.append("files", f);
  form.append("folder", folder);

  const r = await fetch("/upload", { method: "POST", body: form });
  const msg = document.getElementById("uploadMsg");
  if (!r.ok) {
    msg.textContent = "Upload failed";
    return;
  }
  const data = await r.json();
  const n = data.saved?.length || 0;
  const added = data.saved?.reduce((acc, s) => acc + (s.ingest?.added || 0), 0);
  msg.textContent = `Uploaded ${n} file(s). Indexed ${added} chunks.`;
  await refreshDocs();
}

async function search() {
  const q = (document.getElementById("searchInput").value || "").trim();
  if (!q) return;
  const r = await fetch(`/rag/search?q=${encodeURIComponent(q)}&k=5`);
  const data = await r.json();
  const out = document.getElementById("searchOut");
  out.innerHTML = "";
  if (!data.results || data.results.length === 0) {
    out.textContent = data.note || "No results.";
    return;
  }
  for (const res of data.results) {
    const box = document.createElement("div");
    box.className = "p-3 rounded border bg-slate-50";
    const src = document.createElement("a");
    src.href = res.source;
    src.target = "_blank";
    src.className = "text-blue-600 underline";
    src.textContent = `${res.folder}/${res.filename}#chunk${res.chunk_index}`;
    const p = document.createElement("p");
    p.className = "mt-1";
    p.textContent = res.text;
    const s = document.createElement("div");
    s.className = "text-xs text-slate-500 mt-1";
    s.textContent = `score: ${res.score.toFixed(3)}`;
    box.appendChild(src);
    box.appendChild(p);
    box.appendChild(s);
    out.appendChild(box);
  }
}

// Wire UI
document.getElementById("btnStart").addEventListener("click", startVoice);
document.getElementById("btnStop").addEventListener("click", stopVoice);
document.getElementById("btnUpload").addEventListener("click", uploadFiles);
document.getElementById("btnSearch").addEventListener("click", search);

refreshDocs();
```

**requirements.txt**

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
httpx>=0.27.0
python-multipart>=0.0.9

# Stage 02 RAG
faiss-cpu>=1.8.0
numpy>=1.26.0
tiktoken>=0.7.0
PyMuPDF>=1.24.7
python-docx>=1.0.1
```

**Dockerfile**

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# faiss-cpu needs libgomp on Debian slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
COPY data ./data

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**.gitignore**

```
__pycache__/
*.pyc
.env
data/docs/*
!data/.gitkeep
data/index.faiss
data/index_meta.json
```

**.env.example**

```
# Required
OPENAI_API_KEY=sk-your-key

# Realtime
REALTIME_MODEL=gpt-4o-realtime-preview-2025-06-03
REALTIME_VOICE=verse

# Embeddings
EMBEDDING_MODEL=text-embedding-3-small
# Optional chunking
# CHUNK_TOKENS=400
# CHUNK_OVERLAP=40
```

---

**Run instructions**

1. Set your key.

   * Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Local run.

   * `pip install -r requirements.txt`
   * `uvicorn app.main:app --reload`
   * Open `http://localhost:8000/`.
3. Docker run (recommended for FAISS).

   * `docker build -t ai-dental-voice .`
   * `docker run -p 8000:8000 --env-file .env ai-dental-voice`
4. Test the flow.

   * Upload a PDF, DOCX, or TXT. Optionally set a folder name.
   * After upload, chunks are embedded and indexed.
   * Use “RAG Search” to query. You get top‑k snippets with source URLs.
5. Notes.

   * Index files live under `data/index.faiss` and `data/index_meta.json`.
   * Re‑uploading adds to the same index. No delete yet. Rebuild is out of scope for Stage 02.
