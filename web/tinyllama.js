/*
 * tinyllama.js - API cliente para el motor WASM de TinyLlama-1.1B
 *
 * Uso:
 *   import { TinyLlama } from './tinyllama.js';
 *   const llm = await TinyLlama.load({ onProgress: (f) => console.log(f) });
 *   for await (const text of llm.generate('The secret to happiness is', { maxTokens: 64 })) {
 *     process.stdout.write(text);
 *   }
 *
 * El modelo (tinillama.gguf) esta troceado en web/model/tinillama.gguf.NN
 * porque GitHub Pages no sirve archivos > 100 MB. Se descargan en paralelo,
 * se reensamblan en memoria WASM y los pesos Q4_0 se usan sin copia.
 */

import createModule from './tinyllama-engine.js';

const DEFAULTS = {
  modelBase: 'model/',     // donde viven manifest.json y los trozos
  concurrency: 4,          // descargas paralelas de trozos
};

export class TinyLlama {
  #module;
  #modelPtr;

  constructor(module, modelPtr) {
    this.#module = module;
    this.#modelPtr = modelPtr;
  }

  /**
   * Descarga los trozos del modelo, los reensambla en memoria WASM y
   * carga el motor. onProgress recibe la fraccion [0,1] descargada.
   */
  static async load({ modelBase = DEFAULTS.modelBase, concurrency = DEFAULTS.concurrency, onProgress } = {}) {
    const manifest = await (await fetch(modelBase + 'manifest.json')).json();
    const total = manifest.size;
    const parts = manifest.parts;

    // buffer destino en el heap JS
    const bytes = new Uint8Array(total);
    let downloaded = 0;

    // offsets de cada trozo
    let off = 0;
    const offsets = parts.map(() => 0); // se rellena tras conocer tamanos

    // descarga un trozo con su indice
    const fetchPart = async (i) => {
      const res = await fetch(modelBase + parts[i]);
      if (!res.ok) throw new Error(`HTTP ${res.status} en ${parts[i]}`);
      const buf = new Uint8Array(await res.arrayBuffer());
      return { i, buf };
    };

    // pool de concurrencia
    const results = new Array(parts.length);
    let next = 0;
    const worker = async () => {
      while (next < parts.length) {
        const i = next++;
        const { buf } = await fetchPart(i);
        results[i] = buf;
        downloaded += buf.length;
        if (onProgress) onProgress(Math.min(1, downloaded / total));
      }
    };
    await Promise.all(Array.from({ length: concurrency }, worker));

    // reensamblar
    let pos = 0;
    for (const buf of results) {
      bytes.set(buf, pos);
      pos += buf.length;
    }
    if (pos !== total) throw new Error(`Tamano incorrecto: ${pos} != ${total}`);

    // crear modulo WASM y copiar el modelo a su memoria
    const module = await createModule();
    const ptr = module._malloc(total);
    module.HEAPU8.set(bytes, ptr);
    module._tl_init(ptr, total);

    return new TinyLlama(module, ptr);
  }

  /**
   * Genera texto. Devuelve un async generator que va soltando trozos
   * de texto (streaming). Opciones: maxTokens, temp (0 = greedy), topK.
   */
  async *generate(prompt, { maxTokens = 128, temp = 0.7, topK = 40 } = {}) {
    const M = this.#module;
    M.ccall('tl_set_prompt', 'number', ['string'], [prompt]);
    for (let i = 0; i < maxTokens; i++) {
      const id = M._tl_step(temp, topK);
      if (id < 0) break;
      yield M.UTF8ToString(M._tl_token_str(id));
      // ceder el hilo para que la UI respire entre tokens
      await new Promise((r) => setTimeout(r, 0));
    }
  }

  /** Atajo: genera y devuelve el texto completo. */
  async complete(prompt, opts = {}) {
    let out = '';
    for await (const t of this.generate(prompt, opts)) out += t;
    return out;
  }
}

export default TinyLlama;
