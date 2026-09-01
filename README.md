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

---

# Modelo adaptativo (adaptive/)

LLM propio byte/BPE-level que **se entrena solo** en GitHub Actions y mejora
día a día. Documento completo del plan: [`roadmap.md`](roadmap.md).

## Evolución medida (PPL de validación)

| Fecha | Optimizador | Steps acumulados | PPL val | Notas |
|---|---|---|---|---|
| 08-30 | AdamW | 300 | 15.23 | primer ciclo (byte-level) |
| 08-30 | AdamW | 1500 | 18.2→16.2 | corpus peng-mimo |
| 08-30 | AdamW | 4500 | 13.7 | + FineTome-100k rotante |
| 08-30 | AdamW | 12000 | **5.43** | BPE vocab 2048, dim 192×6 |
| — | MuonH A/B | 2000 | 5.45 | Adam ganó sobre modelo calentado |
| — | (próximas) | +9000/ciclo ×8/día | ↓ | WSD + CMA + expansión Net2Net |

## Pipeline automático

```
peng-mimo nocturno (código verificado)  ┐
FineTome-100k (rotante 15MB)            ├→ Actions 8×/día → gate PPL → commit
open-perfectblend (rotante 10MB)        ┘        ↓
checkpoint del PC (watcher cada 10 min) ┘        Pages redeploy
        ↓
chat web (WASM 2MB): métricas + histórico + entrenar en vivo + contribuir
```

- Repositorio de datos: [`fivetechsoft/peng-mimo`](https://github.com/FiveTechSoft/peng-mimo)
- El **gate de PPL** es el árbitro: solo se publica si la perplexity de
  validación mejora (nunca se degrada el modelo público)
- **Saturation → auto-expansión Net2Net** (dim 192→384→768): el `ad_expand`
  duplica capacidad preservando la función aprendida
