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


