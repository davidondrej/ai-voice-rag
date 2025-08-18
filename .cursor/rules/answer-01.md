Here is the complete monolith scaffold. It uses WebRTC with an ephemeral `/v1/realtime/sessions` token as per OpenAI’s Realtime docs. ([OpenAI Platform][1])

**File tree**

```
ai-dental-voice/
  app/
    main.py
  static/
    index.html
    app.js
  data/
    .gitkeep
  .env.example
  .gitignore
  Dockerfile
  requirements.txt
```

**app/main.py**

```python
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
      <p class="text-sm text-slate-600">WebRTC voice agent + uploads. Stage 1 scaffold.</p>
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
      <input id="inputFiles" type="file" accept=".pdf,.docx,.txt" multiple class="block" />
      <button id="btnUpload" class="px-4 py-2 rounded bg-black text-white">Upload</button>
      <div id="uploadMsg" class="text-sm text-slate-600"></div>
    </section>

    <section class="p-4 bg-white rounded-lg border space-y-4">
      <h2 class="font-medium">Documents</h2>
      <div id="docsList" class="space-y-2 text-sm"></div>
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
    // 1) Get ephemeral key from backend
    const r = await fetch("/session");
    if (!r.ok) {
      const t = await r.text();
      throw new Error("failed /session: " + t);
    }
    const data = await r.json();
    const EPHEMERAL_KEY = data?.client_secret?.value;
    if (!EPHEMERAL_KEY) throw new Error("no ephemeral key");

    // 2) Create PeerConnection
    pc = new RTCPeerConnection();
    pc.onconnectionstatechange = () => {
      connEl.textContent = pc.connectionState;
    };

    // 3) Remote audio
    pc.ontrack = (e) => {
      remoteAudio.srcObject = e.streams[0];
    };

    // 4) Local mic
    localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    micEl.textContent = "on";
    pc.addTrack(localStream.getTracks()[0]);

    // 5) Data channel for events/logs
    dc = pc.createDataChannel("oai-events");
    dc.onmessage = (e) => log("[event] " + e.data);

    // 6) SDP offer -> OpenAI Realtime over WebRTC
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const baseUrl = "https://api.openai.com/v1/realtime";
    const model = "gpt-4o-realtime-preview-2025-06-03";
    const sdpResp = await fetch(`${baseUrl}?model=${encodeURIComponent(model)}`, {
      method: "POST",
      body: offer.sdp,
      headers: {
        Authorization: `Bearer ${EPHEMERAL_KEY}`,
        "Content-Type": "application/sdp",
      },
    });

    if (!sdpResp.ok) {
      const txt = await sdpResp.text();
      throw new Error("SDP exchange failed: " + txt);
    }

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
    if (localStream) {
      localStream.getTracks().forEach(t => t.stop());
      localStream = null;
    }
  } finally {
    micEl.textContent = "off";
    connEl.textContent = "disconnected";
    statusEl.textContent = "idle";
  }
}

// Uploads and listing
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
    a.textContent = `${d.name} (${d.size} bytes)`;
    a.className = "text-blue-600 underline";
    a.target = "_blank";
    const div = document.createElement("div");
    div.appendChild(a);
    list.appendChild(div);
  }
}

async function uploadFiles() {
  const files = document.getElementById("inputFiles").files;
  if (!files || files.length === 0) return;
  const form = new FormData();
  for (const f of files) form.append("files", f);
  const r = await fetch("/upload", { method: "POST", body: form });
  const msg = document.getElementById("uploadMsg");
  if (!r.ok) {
    msg.textContent = "Upload failed";
    return;
  }
  const data = await r.json();
  msg.textContent = `Uploaded ${data.saved?.length || 0} file(s).`;
  await refreshDocs();
}

document.getElementById("btnStart").addEventListener("click", startVoice);
document.getElementById("btnStop").addEventListener("click", stopVoice);
document.getElementById("btnUpload").addEventListener("click", uploadFiles);

refreshDocs();
```

**requirements.txt**

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
httpx>=0.27.0
python-multipart>=0.0.9
```

**Dockerfile**

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
COPY data ./data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**.env.example**

```
# Required
OPENAI_API_KEY=sk-your-key

# Optional
REALTIME_MODEL=gpt-4o-realtime-preview-2025-06-03
REALTIME_VOICE=verse
```

**.gitignore**

```
__pycache__/
*.pyc
.env
data/docs/*
!data/.gitkeep
```

**data/.gitkeep**

```
```

Run locally:

* `pip install -r requirements.txt`
* `OPENAI_API_KEY=... uvicorn app.main:app --reload`

Open `http://localhost:8000/`.

[1]: https://platform.openai.com/docs/guides/realtime?utm_source=chatgpt.com "Realtime API - OpenAI Platform"
