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


