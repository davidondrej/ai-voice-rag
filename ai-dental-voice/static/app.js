// Uses OpenAI Agents SDK for Realtime + a single client-side tool.
// CDN ESM for quick prototype (no build step).
import {
  RealtimeAgent,
  RealtimeSession,
  OpenAIRealtimeWebRTC,
  tool,
} from 'https://cdn.jsdelivr.net/npm/@openai/agents-realtime/+esm';
import { z } from 'https://cdn.jsdelivr.net/npm/zod/+esm';

let session = null;
let transport = null;
let agent = null;
let localStream = null;

const MODEL = 'gpt-4o-realtime-preview-2025-06-03'; // keep consistent with backend
const VOICE = 'verse';

const logEl = document.getElementById('log');
const statusEl = document.getElementById('status');
const micEl = document.getElementById('micState');
const connEl = document.getElementById('connState');
const remoteAudio = document.getElementById('remoteAudio');
const transcriptEl = document.getElementById('transcript');
const answerEl = document.getElementById('answer');

function log(s) {
  logEl.textContent += s + '\n';
  logEl.scrollTop = logEl.scrollHeight;
}

function resetDisplays() {
  transcriptEl.textContent = '';
  answerEl.textContent = '';
}

const instructions = [
  'You are a dental clinic voice agent.',
  'Answer only using the clinic knowledge base accessed via the search_kb tool.',
  'Always call search_kb(query) before answering.',
  'Cite sources in speech: say "Source: <filename>#chunkN".',
  'If the KB lacks info, ask the user to upload relevant documents and specify what is missing.',
  'Keep answers concise and conversational.',
].join('\n');

// Single tool: search_kb(query) -> calls backend /rag/search and returns snippets + citations.
const searchKb = tool({
  name: 'search_kb',
  description: 'Search the clinic KB for facts and citations. Use before answering.',
  parameters: z.object({ query: z.string() }),
  // Return structured JSON the model can consume.
  async execute({ query }) {
    const r = await fetch(`/rag/search?q=${encodeURIComponent(query)}&k=5`);
    const data = await r.json();
    const results = (data.results || []).map((x) => ({
      text: x.text,
      source: x.source,
      citation: `${x.folder}/${x.filename}#chunk${x.chunk_index}`,
      score: x.score,
    }));
    return { results };
  },
});

// Voice control
async function startVoice() {
  if (session) return;
  statusEl.textContent = 'starting';
  connEl.textContent = 'connecting';
  resetDisplays();

  try {
    // 1) Mint ephemeral key from backend
    const r = await fetch('/session');
    if (!r.ok) throw new Error('failed /session: ' + (await r.text()));
    const data = await r.json();
    const EPHEMERAL_KEY = data?.client_secret?.value;
    if (!EPHEMERAL_KEY) throw new Error('no ephemeral key');

    // 2) Prepare mic and output audio
    localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    micEl.textContent = 'on';

    transport = new OpenAIRealtimeWebRTC({
      mediaStream: localStream,
      audioElement: remoteAudio,
    });

    // 3) Define agent with tool and instructions
    agent = new RealtimeAgent({
      name: 'DentalKB',
      instructions,
      tools: [searchKb],
    });

    // 4) Create session and connect with WebRTC
    session = new RealtimeSession(agent, {
      transport,
      model: MODEL,
    });

    // 5) Wire transport events for transcript and status
    session.transport.on('connection_change', (state) => {
      connEl.textContent = state;
    });

    // Stream assistant audio transcript deltas to "Last answer"
    let answerBuf = '';
    session.transport.on('audio_transcript_delta', (evt) => {
      if (!evt || typeof evt.delta !== 'string') return;
      answerBuf += evt.delta;
      answerEl.textContent = answerBuf;
    });
    session.transport.on('audio_done', () => {
      answerBuf += '\n';
      answerEl.textContent = answerBuf;
    });

    // Optional: log tool calls and errors for debugging
    session.transport.on('function_call', (evt) => {
      try {
        const name = evt?.name || 'tool';
        const args = evt?.arguments ? JSON.stringify(evt.arguments) : '';
        log(`[tool] ${name} ${args}`);
      } catch (_) {}
    });
    session.transport.on('error', (e) => {
      log('ERROR: ' + JSON.stringify(e));
    });

    // 6) Connect with ephemeral token. Voice configured server-side too.
    await session.connect({
      apiKey: EPHEMERAL_KEY,
      initialSessionConfig: {
        voice: VOICE,
        modalities: ['text', 'audio'],
      },
    });

    statusEl.textContent = 'connected';
    log('voice agent ready');
  } catch (err) {
    statusEl.textContent = 'error';
    log('ERROR: ' + (err?.message || String(err)));
    await stopVoice();
  }
}

async function stopVoice() {
  try {
    if (session) {
      await session.disconnect();
      session = null;
    }
    if (transport?.close) {
      try { transport.close(); } catch (_) {}
      transport = null;
    }
    if (localStream) {
      localStream.getTracks().forEach((t) => t.stop());
      localStream = null;
    }
  } finally {
    micEl.textContent = 'off';
    connEl.textContent = 'disconnected';
    statusEl.textContent = 'idle';
  }
}

// ---- Uploads, listing, and search UI (kept from Stage 02) ----
async function refreshDocs() {
  const r = await fetch('/list');
  const data = await r.json();
  const list = document.getElementById('docsList');
  list.innerHTML = '';
  if (!data.docs || data.docs.length === 0) {
    list.textContent = 'No documents uploaded yet.';
    return;
  }
  for (const d of data.docs) {
    const a = document.createElement('a');
    a.href = d.url;
    a.textContent = `${d.folder}/${d.name} (${d.size} bytes)`;
    a.className = 'text-blue-600 underline';
    a.target = '_blank';
    const div = document.createElement('div');
    div.appendChild(a);
    list.appendChild(div);
  }
}

async function uploadFiles() {
  const files = document.getElementById('inputFiles').files;
  const folder = (document.getElementById('folderInput').value || 'default').trim();
  if (!files || files.length === 0) return;
  const form = new FormData();
  for (const f of files) form.append('files', f);
  form.append('folder', folder);

  const r = await fetch('/upload', { method: 'POST', body: form });
  const msg = document.getElementById('uploadMsg');
  if (!r.ok) {
    msg.textContent = 'Upload failed';
    return;
  }
  const data = await r.json();
  const n = data.saved?.length || 0;
  const added = data.saved?.reduce((acc, s) => acc + (s.ingest?.added || 0), 0);
  msg.textContent = `Uploaded ${n} file(s). Indexed ${added} chunks.`;
  await refreshDocs();
}

async function search() {
  const q = (document.getElementById('searchInput').value || '').trim();
  if (!q) return;
  const r = await fetch(`/rag/search?q=${encodeURIComponent(q)}&k=5`);
  const data = await r.json();
  const out = document.getElementById('searchOut');
  out.innerHTML = '';
  if (!data.results || data.results.length === 0) {
    out.textContent = data.note || 'No results.';
    return;
  }
  for (const res of data.results) {
    const box = document.createElement('div');
    box.className = 'p-3 rounded border bg-slate-50';
    const src = document.createElement('a');
    src.href = res.source;
    src.target = '_blank';
    src.className = 'text-blue-600 underline break-all';
    src.textContent = `${res.folder}/${res.filename}#chunk${res.chunk_index}`;
    const p = document.createElement('p');
    p.className = 'mt-1';
    p.textContent = res.text;
    const s = document.createElement('div');
    s.className = 'text-xs text-slate-500 mt-1';
    s.textContent = `score: ${res.score.toFixed(3)}`;
    box.appendChild(src);
    box.appendChild(p);
    box.appendChild(s);
    out.appendChild(box);
  }
}

// Wire UI
document.getElementById('btnStart').addEventListener('click', startVoice);
document.getElementById('btnStop').addEventListener('click', stopVoice);
document.getElementById('btnUpload').addEventListener('click', uploadFiles);
document.getElementById('btnSearch').addEventListener('click', search);

refreshDocs();


