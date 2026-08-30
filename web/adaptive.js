/*
 * adaptive.js - API cliente del modelo adaptativo para el navegador.
 *
 *   import { Adaptive } from './adaptive.js';
 *   const llm = await Adaptive.load({ onProgress, onMetadata });
 *   for await (const t of llm.generate(prompt, opts)) ...
 */

const DEFAULTS = {
  modelBase: 'adaptive/',
  workerUrl: 'adaptive-worker.js',
};

export class Adaptive {
  #worker;
  #seq = 0;

  constructor(worker) {
    this.#worker = worker;
  }

  static load({ modelBase = DEFAULTS.modelBase, workerUrl = DEFAULTS.workerUrl,
                onProgress, onMetadata } = {}) {
    const worker = new Worker(workerUrl, { type: 'module' });
    return new Promise((resolve, reject) => {
      const onMsg = (e) => {
        const m = e.data;
        if (m.type === 'progress' && onProgress) onProgress(m.f);
        else if (m.type === 'metadata' && onMetadata) onMetadata(m);
        else if (m.type === 'ready') {
          worker.removeEventListener('message', onMsg);
          resolve(new Adaptive(worker));
        } else if (m.type === 'error') {
          worker.removeEventListener('message', onMsg);
          reject(new Error(m.message));
        }
      };
      worker.addEventListener('message', onMsg);
      worker.addEventListener('error', (ev) => reject(new Error(ev.message)));
      worker.postMessage({ cmd: 'load', modelBase });
    });
  }

  /** atajo para la UI de entrenamiento en vivo */
  postMessage(msg, transfer) { this.#worker.postMessage(msg, transfer); }
  addEventListener(type, h) { this.#worker.addEventListener(type, h); }
  removeEventListener(type, h) { this.#worker.removeEventListener(type, h); }

  async *generate(prompt, { maxTokens = 128, temp = 0.8, topK = 40 } = {}) {
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
    this.#worker.postMessage({ cmd: 'gen', id, prompt, maxTokens, temp, topK });
    try {
      for (;;) {
        while (queue.length) {
          const m = queue.shift();
          if (m.type === 'token') yield m.text;
          else if (m.type === 'prefillDone') yield { prefillSeconds: m.seconds ?? 0 };
          else if (m.type === 'done') return;
          else if (m.type === 'error') throw new Error(m.message);
        }
        await new Promise((r) => { wake = r; });
      }
    } finally {
      this.#worker.removeEventListener('message', onMsg);
    }
  }
}

export default Adaptive;