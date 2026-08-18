# tinyllama

TinyLlama-1.1B (Q4_0) ejecutándose **íntegramente en el navegador**: motor de
inferencia en C puro compilado a WebAssembly, servido como sitio estático en
GitHub Pages. Sin servidor, sin backend.

**Demo:** https://fivetechsoft.github.io/tinyllama/

## Cómo funciona

- `engine/tinyllama.c` — motor de inferencia C puro (sin dependencias), basado
  en [`llm_inference.c` de fivetechsoft/dreaming](https://github.com/fivetechsoft/dreaming).
  Adaptado para desquantizar Q4_0 on-the-fly (el modelo ocupa ~1 GB de RAM en
  vez de 4,4 GB) y con exports para Emscripten (`tl_init`, `tl_set_prompt`,
  `tl_step`, `tl_token_str`).
- `web/model/tinillama.gguf.NN` — el modelo
  [TinyLlama-1.1B-Chat Q4_0](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)
  (608 MB) troceado en partes de 50 MB, porque GitHub Pages no sirve archivos
  de más de 100 MB. El navegador los descarga en paralelo y los reensambla en
  memoria WASM (`web/model/manifest.json` describe las partes).
- `web/tinyllama.js` — API JavaScript cliente (ver abajo).
- `.github/workflows/deploy.yml` — GitHub Action que compila el motor a WASM
  con Emscripten y despliega el sitio a GitHub Pages.

## API de uso

```js
import { TinyLlama } from './tinyllama.js';

// Descarga el modelo (608 MB la primera vez) y lo carga en WASM
const llm = await TinyLlama.load({
  onProgress: (f) => console.log((f * 100).toFixed(0) + '%'),
});

// Generación con streaming (async generator)
for await (const text of llm.generate('The secret to happiness is', {
  maxTokens: 64,
  temp: 0.7,   // 0 = greedy
  topK: 40,
})) {
  process.stdout.write(text);
}

// Atajo sin streaming
const answer = await llm.complete('The secret to happiness is', { maxTokens: 64 });
```

## Uso nativo (depuración)

```sh
gcc -O2 -march=native -o tinyllama engine/tinyllama.c -lm
./tinyllama model/tinillama.gguf "The secret to happiness is" 60 0.7 40
#                modelo                  prompt                 n  temp topk
```

## Limitaciones conocidas

- Single-thread WASM (GitHub Pages no permite las cabeceras COOP/COEP
  necesarias para `SharedArrayBuffer`); SIMD128 sí está activo.
- Velocidad: ~1-3 tok/s según máquina (es un motor educativo, no llama.cpp).
- La descarga inicial del modelo (608 MB) tarda según la conexión; el
  navegador cachea los trozos en siguientes visitas.

## Licencia

Código abierto, como el proyecto [dreaming](https://github.com/fivetechsoft/dreaming)
del que deriva el motor. El modelo TinyLlama tiene su propia licencia (Apache 2.0).
