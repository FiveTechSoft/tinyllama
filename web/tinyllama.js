/*
 * tinyllama.js - API cliente para TinyLlama-1.1B en el navegador.
 *
 * La inferencia corre en un Web Worker (web/llm-worker.js) para no congelar
 * la UI; esta clase es una envoltura con la misma API de siempre:
 *
 *   import { TinyLlama } from './tinyllama.js';
 *   const llm = await TinyLlama.load({ onProgress: (f) => ... });
 *   for await (const text of llm.generate(prompt, { maxTokens: 64, temp: 0.7, topK: 40 })) ...
 */

const DEFAULTS = {
  modelBase: 'model/',
  concurrency: 4,
  workerUrl: 'llm-worker.js',
};

export class TinyLlama {
  #worker;
  #seq = 0;

  constructor(worker) {
    this.#worker = worker;
  }

  /** Lanza el worker, descarga el modelo y carga el motor WASM. */
  static load({ modelBase = DEFAULTS.modelBase, concurrency = DEFAULTS.concurrency,
                workerUrl = DEFAULTS.workerUrl, onProgress } = {}) {
    const worker = new Worker(workerUrl, { type: 'module' });
    return new Promise((resolve, reject) => {
      const onMsg = (e) => {
        const m = e.data;
        if (m.type === 'progress' && onProgress) onProgress(m.f);
        else if (m.type === 'ready') {
          worker.removeEventListener('message', onMsg);
          resolve(new TinyLlama(worker));
        } else if (m.type === 'error') {
          worker.removeEventListener('message', onMsg);
          reject(new Error(m.message));
        }
      };
      worker.addEventListener('message', onMsg);
      worker.addEventListener('error', (ev) => reject(new Error(ev.message)));
      worker.postMessage({ cmd: 'load', modelBase, concurrency });
    });
  }

  /**
   * Genera texto (async generator, streaming). La generacion ocurre en el
   * worker; salir del bucle con break la cancela.
   */
  async *generate(prompt, { maxTokens = 128, temp = 0.7, topK = 40 } = {}) {
    const id = ++this.#seq;
    const queue = [];
    let wake = null;
    const onMsg = (e) => {
      const m = e.data;
      if (m.id !== id) return;
      queue.push(m);
      if (wake) { wake(); wake = null; }
    };
    this.#worker.addEventListener('message', onMsg);
    this.#worker.postMessage({ cmd: 'generate', id, prompt, maxTokens, temp, topK });
    try {
      for (;;) {
        while (queue.length) {
          const m = queue.shift();
          if (m.type === 'token') yield m.text;
          else if (m.type === 'prefillDone') yield { prefillSeconds: m.seconds };
          else if (m.type === 'done') return;
          else if (m.type === 'error') throw new Error(m.message);
        }
        await new Promise((r) => { wake = r; });
      }
    } finally {
      this.#worker.postMessage({ cmd: 'cancel', id });
      this.#worker.removeEventListener('message', onMsg);
    }
  }

  /** Atajo: genera y devuelve el texto completo. */
  async complete(prompt, opts = {}) {
    let out = '';
    for await (const t of this.generate(prompt, opts)) {
      if (typeof t === 'string') out += t;
    }
    return out;
  }
}

export default TinyLlama;
