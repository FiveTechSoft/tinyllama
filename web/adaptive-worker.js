/*
 * adaptive-worker.js - Web Worker del modelo adaptativo (.adm, ~2MB)
 * FAST PATH: generation batch en UNA llamada WASM (gigakernel) y
 * streaming de bloques a la UI (~8 tokens por frame) en vez de 1 a 1.
 */
import createAdaptive from './adaptive-engine.js';

let M = null;

async function loadModel(url, post) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} en ${url}`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  post({ type: 'progress', f: 1 });
  M = await createAdaptive();
  const ptr = M._malloc(bytes.length);
  M.HEAPU8.set(bytes, ptr);
  const r = M._ad_load_mem(ptr, bytes.length);
  M._free(ptr);
  if (r !== 0) throw new Error(`ad_load_mem = ${r}`);
  const st = M._malloc(8 * 4);
  M._ad_stats(st, 8);
  const s = new Int32Array(M.HEAPF32.buffer, st, 8);
  post({
    type: 'metadata',
    params: s[0], dim: s[1], layers: s[2], heads: s[3],
    hidden: s[4], seq: s[5], steps: s[6],
    sizeMB: (bytes.length / 1048576).toFixed(2),
    mp: (s[0] / 1e6).toFixed(2),
  });
  M._free(st);
}

self.onmessage = async (e) => {
  const m = e.data;
  const post = (x) => self.postMessage(x);
  try {
    if (m.cmd === 'load') {
      await loadModel(m.url ?? 'adaptive/model.adm', post);
      post({ type: 'ready' });
    } else if (m.cmd === 'gen') {
      const maxT = Math.min(m.maxTokens || 200, 400);
      const bufPtr = M._malloc(maxT);
      const t0 = performance.now();
      const n = M.ccall(
        'ad_generate_n', 'number',
        ['string', 'number', 'number', 'number', 'number'],
        [m.prompt, maxT, m.temp, m.topK, bufPtr],
      );
      /* streaming por bloques: la UI ve texto fluido sin dispatch por token */
      const chunkSize = 4;
      for (let i = 0; i < n; i += chunkSize) {
        const end = Math.min(i + chunkSize, n);
        let s = '';
        for (let k = i; k < end; k++) s += String.fromCharCode(M.HEAPU8[bufPtr + k]);
        post({ type: 'token', id: m.id, text: s });
        await new Promise((r) => setTimeout(r, 0));
      }
      M._free(bufPtr);
      post({ type: 'done', id: m.id, tokens: n });
    }
  } catch (err) {
    post({ type: 'error', id: m.id, message: String((err && err.message) || err) });
  }
};