import os
from pathlib import Path
from typing import List

import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- Config ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
REALTIME_MODEL = os.environ.get("REALTIME_MODEL", "gpt-4o-realtime-preview-2025-06-03")
REALTIME_VOICE = os.environ.get("REALTIME_VOICE", "verse")

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"

DOCS_DIR.mkdir(parents=True, exist_ok=True)

# --- App ---
app = FastAPI(title="AI Dental Voice - Monolith Scaffold")

# CORS (safe default for single-origin local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI and uploaded files
app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")
app.mount("/files", StaticFiles(directory=DATA_DIR, html=False), name="files")

# Root -> index.html
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = STATIC_DIR / "index.html"
    return index_path.read_text(encoding="utf-8")


# 1) Mint ephemeral Realtime token for WebRTC
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
    # Return upstream JSON and status to the browser
    return JSONResponse(status_code=resp.status_code, content=resp.json())


# 2) Upload documents (PDF, DOCX, TXT). Saved under data/docs/
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt"}

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    saved = []
    for f in files:
        name = Path(f.filename).name  # basic sanitization
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

        dest = DOCS_DIR / name
        # Prevent accidental overwrite: add numeric suffix if exists
        if dest.exists():
            stem = dest.stem
            suffix_txt = dest.suffix
            i = 1
            while True:
                candidate = DOCS_DIR / f"{stem} ({i}){suffix_txt}"
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

        saved.append({
            "name": dest.name,
            "size": dest.stat().st_size,
            "url": f"/files/docs/{dest.name}",
        })

    return {"saved": saved}


# 3) List uploaded documents
@app.get("/list")
async def list_docs():
    items = []
    for p in sorted(DOCS_DIR.glob("*")):
        if p.is_file():
            items.append({
                "name": p.name,
                "size": p.stat().st_size,
                "url": f"/files/docs/{p.name}",
            })
    return {"docs": items}


# 4) RAG search placeholder (wired for step 2 later)
@app.get("/rag/search")
async def rag_search(q: str):
    # Placeholder for stage 1
    return {
        "query": q,
        "results": [],
        "note": "RAG not implemented in stage 1. Endpoint exists for wiring."
    }


