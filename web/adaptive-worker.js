/*
 * adaptive-worker.js - Web Worker del modelo adaptativo (.adm, ~1-2MB)
 * Carga instantanea + metricas de mejora del modelo en cada sesion.
 *
 * Mensajes: {cmd:'load', url} {cmd:'gen', id, prompt, maxTokens, temp, topK}
 * Sale:     {type:'progress'|'ready'|'metadata'|'token'|'done'|'error'}
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
  /* metricas: [params, dim, layers, heads, hidden, seq, steps, fmtver] */
  const st = M._malloc(8 * 4);
  M._ad_stats(st, 8);
  const s = new Int32Array(M.HEAPF32.buffer, st, 8);
  post({
    type: 'metadata',
    params: s[0],
    dim: s[1], layers: s[2], heads: s[3], hidden: s[4], seq: s[5],
    steps: s[6],
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
      M._ad_set_prompt_mem(m.prompt);
      post({ type: 'prefillDone', id: m.id });
      for (let i = 0; i < m.maxTokens; i++) {
        const id = M._ad_step_mem(m.temp, m.topK);
        if (id < 0) break;
        const t = M._ad_tok_str(id);
        post({ type: 'token', id: m.id, text: M.UTF8ToString(t) });
        await new Promise((r) => setTimeout(r, 0));
      }
      post({ type: 'done', id: m.id });
    }
  } catch (err) {
    post({ type: 'error', id: m.id, message: String((err && err.message) || err) });
  }
};