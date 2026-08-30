/*
 * adaptive-worker.js - Web Worker del modelo adaptativo (.adm, ~2MB)
 * FAST PATH: generation batch en UNA llamada WASM (gigakernel) y
 * streaming de bloques a la UI (~8 tokens por frame) en vez de 1 a 1.
 */
import createAdaptive from './adaptive-engine.js';

let M = null;
let modelBytes = null;   /* buffer original del .adm para ad_live_start */

async function loadModel(url, post) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} en ${url}`);
  modelBytes = new Uint8Array(await res.arrayBuffer());
  post({ type: 'progress', f: 1 });
  M = await createAdaptive();
  const ptr = M._malloc(modelBytes.length);
  M.HEAPU8.set(modelBytes, ptr);
  const r = M._ad_load_mem(ptr, modelBytes.length);
  M._free(ptr);
  if (r !== 0) throw new Error(`ad_load_mem = ${r}`);
  const st = M._malloc(8 * 4);
  M._ad_stats(st, 8);
  const s = new Int32Array(M.HEAPF32.buffer, st, 8);
  post({
    type: 'metadata',
    params: s[0], dim: s[1], layers: s[2], heads: s[3],
    hidden: s[4], seq: s[5], steps: s[6],
    sizeMB: (modelBytes.length / 1048576).toFixed(2),
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
    } else if (m.cmd === 'live_start') {
      /* corpus para entrenar en vivo; el modelo original (modelBytes)
         se conserva de loadModel() y se pasa a ad_live_start */
      const corpusBytes = new TextEncoder().encode(m.corpus);
      const cPtr = M._malloc(corpusBytes.length);
      M.HEAPU8.set(corpusBytes, cPtr);
      if (!modelBytes) throw new Error('modelo no residente');
      const mPtr = M._malloc(modelBytes.length);
      M.HEAPU8.set(modelBytes, mPtr);
      const r = M._ad_live_start(mPtr, modelBytes.length, cPtr,
                                 corpusBytes.length, m.ctx || 96);
      M._free(cPtr);
      post({ type: 'liveStarted', id: m.id, r });
    } else if (m.cmd === 'live_chunk') {
      const loss = M._ad_live_chunk(m.steps, m.batch, m.lr);
      const ppl = M._ad_live_eval();
      const steps = M._ad_live_steps();
      post({ type: 'liveProgress', id: m.id, loss, ppl, steps });
    } else if (m.cmd === 'live_save') {
      const need = M._ad_live_save(0, 0);         /* hack: devuelve -2 sin buffer */
      const ptr = M._malloc(m.max || 4194304);
      const n = M._ad_live_save(ptr, m.max || 4194304);
      if (n > 0) {
        const bytes = new Uint8Array(M.HEAPU8.subarray(ptr, ptr + n));
        post({ type: 'liveSaved', id: m.id, bytes: bytes }, [bytes.buffer]);
      } else {
        post({ type: 'error', id: m.id, message: 'ad_live_save=' + n });
      }
      M._free();
    }
  } catch (err) {
    post({ type: 'error', id: m.id, message: String((err && err.message) || err) });
  }
};
