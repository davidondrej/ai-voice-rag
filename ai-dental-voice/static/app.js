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


