/*
 * llm-worker.js - Web Worker: carga el motor WASM y el modelo, y genera
 * tokens fuera del hilo principal (la UI nunca se congela).
 *
 * Mensajes entrantes:
 *   {cmd:'load', modelBase}              -> progress/ready/error
 *   {cmd:'generate', id, prompt, maxTokens, temp, topK}
 *   {cmd:'cancel', id}
 * Mensajes salientes:
 *   {type:'progress', f} {type:'ready'}
 *   {type:'token', id, text} {type:'done', id, tokens, seconds}
 *   {type:'error', id?, message}
 */

import createModule from './tinyllama-engine.js';

let M = null;
let cancelled = false;

const post = (m) => self.postMessage(m);
const yield_ = () => new Promise((r) => setTimeout(r, 0));

async function load(modelBase, concurrency) {
  const manifest = await (await fetch(modelBase + 'manifest.json')).json();
  const total = manifest.size;
  const parts = manifest.parts;

  const results = new Array(parts.length);
  let next = 0, downloaded = 0;
  const worker = async () => {
    while (next < parts.length) {
      const i = next++;
      const res = await fetch(modelBase + parts[i]);
      if (!res.ok) throw new Error(`HTTP ${res.status} en ${parts[i]}`);
      results[i] = new Uint8Array(await res.arrayBuffer());
      downloaded += results[i].length;
      post({ type: 'progress', f: Math.min(1, downloaded / total) });
    }
  };
  await Promise.all(Array.from({ length: concurrency }, worker));

  const bytes = new Uint8Array(total);
  let pos = 0;
  for (const b of results) { bytes.set(b, pos); pos += b.length; }
  if (pos !== total) throw new Error(`Tamano incorrecto: ${pos} != ${total}`);

  M = await createModule();
  const ptr = M._malloc(total);
  M.HEAPU8.set(bytes, ptr);
  M._tl_init(ptr, total);
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.cmd === 'load') {
      await load(m.modelBase ?? 'model/', m.concurrency ?? 4);
      post({ type: 'ready' });
    } else if (m.cmd === 'generate') {
      const t0 = performance.now();
      cancelled = false;
      M.ccall('tl_set_prompt', 'number', ['string'], [m.prompt]);
      post({ type: 'prefillDone', id: m.id, seconds: (performance.now() - t0) / 1000 });
      let n = 0;
      for (let i = 0; i < m.maxTokens; i++) {
        if (cancelled) break;
        const id = M._tl_step(m.temp, m.topK);
        if (id < 0) break;
        post({ type: 'token', id: m.id, text: M.UTF8ToString(M._tl_token_str(id)) });
        n++;
        await yield_(); // permite procesar 'cancel'
      }
      post({ type: 'done', id: m.id, tokens: n, seconds: (performance.now() - t0) / 1000 });
    } else if (m.cmd === 'cancel') {
      cancelled = true;
    }
  } catch (err) {
    post({ type: 'error', id: m.id, message: String((err && err.message) || err) });
  }
};
