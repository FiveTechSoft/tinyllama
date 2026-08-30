# Roadmap — modelo adaptativo (tinyllama)

> Mini-LLM byte-level, 100% C, entrenado automáticamente en GitHub Actions
> (4 ciclos/día), servido por GitHub Pages y ejecutado en el navegador vía WASM.
> Objetivo: un modelo público que **mejora solo** cuanto más se usa.

## Estado actual (2026-08-30)

| Componente | Estado | Detalle |
|---|---|---|
| Núcleo C (`adaptive/ad_core.c`) | ✅ prod | formato ADM F32 plano, kernels AVX2 (portados de `dreaming`), KV cache, save/load |
| Trainer (`adaptive/train.c`) | ✅ prod | backprop completo + Adam + cosine schedule + warmup + grad clipping (Karpeles ch.3); gate de publicación por PPL; metrics.csv con params/bytes/dim |
| Mejor PPL lograda | **13.27** | corpus mixto; inicial fue 258 |
| Modelo vivo | ✅ | `adaptive/model.adm` 2MB, dim=96, L=4, H=4, 0.51M pesos |
| Workflow nocturno | ✅ | 4 crons/día: medir→entrenar→medir→commit solo si mejora→re-dispatch Pages |
| Datasets | ✅ | peng-mimo nocturno (C verificado) + FineTome-100k (mlabonne, rotación 15MB/ciclo) + open-perfectblend shard rotante (10MB) — ambos descargados de HF on-the-fly y cacheados |
| Chat nativo (`chat.exe`) | ✅ | chat CLI + grando en `sesion.bin` (aprendizaje de sesión) |
| Chat web (Pages) | ✅ | selector TinyLlama-1.1B/Adaptativo, métricas en vivo, histórico de evolución (curvas PPL/tamaño), entrenamiento EN VIVO en navegador (WASM, loss real) + contribución automática a PR |
| Gate de contribuciones | ✅ | PRs de navegadores evaluadas contra corpus canario; merge solo si PPL mejora |

## Pendientes inmediatos

- [ ] **BPE completo (en curso, fase 1 lista)**: vocab.bpe 2048 aprendido ✅, encoder C ✅ (2.1-2.7 chars/token roundtrip OK), corpora convertidos a ids ✅ (`*_ids.bin`, python encoder verificado), `cfg.vocab` dinámico en el core ✅ (ad_init_fresh_v). **Fase 2 (pendiente):** train.c que consuma ids u32 directamente (hoy su loop es bytes &0xFF) + guardado/decodificación BPE en generación + corpus_math (921KB del mml-book) añadido al mezclador. Los PDFs didácticos (mml-book + repo rohanmistry231) ya aportan corpus matemático.
- [ ] **IndexedDB en la web**: persistir conversaciones + hechos del usuario del chat (el análogo web de `sesion.bin`), hoy la sesión muere al recargar.
- [ ] **Export conversaciones → contrib**: además de pesos, los usuarios podrían contribuir sus conversaciones (con el mismo gate).
- [ ] **int8 en formato ADM**: pesos cuantizados on-disk (2MB→0.5MB, 4× menos descarga); descuantizar on-load; entrena en FP32 (artículo Jalapeño, punto 4).

## Experimentación 30-ago (verificación de comportamiento)

| Modelo | Capacidad | Corpus | PPL | Generación |
|---|---|---|---|---|
| dim96 L4 | 0.55M | mixto (peng+FT+PB) | 13.27 | sopa de palabras |
| dim192 L6 | 2.9M (5.2×) | idem | 13.27 | igual de incoherente |
| dim96 L4 | 0.55M | alpaca pura | 13.49 | sopa + bigramas "the an" |
| dim192 L6 | 2.9M | alpaca pura | 13.80 | igual |

**Conclusión medida:** la capacidad NO era el cuello — 5× capacidad da la misma PPL y mismo nivel de incoherencia. El límite es el tokenizador byte-level (vocab 256). Pasar a BPE ~2k tokens es LA prioridad. PPL byte ~13 ≈ 4-5 bits/byte (cerca de compresión gzip del inglés ~3 bits); el modelo está comprimiendo razonable pero no puede formar patrones de palabra largos con 4 capas.

## Auto-ampliación (Net2Net) — la siguiente escala

Señal de disparo: **PPL de validación estancada 1–2 semanas** con datos nuevos llegando = capacidad llena → expandir.

| Paso | Técnica | Origen |
|---|---|---|
| dim 96→192 | width expansion function-preserving (Net2Net) | paper Net2Net 2015 |
| L 4→8 | depth up-scaling (duplicado de bloques) | SOLAR 10.7B |
| MHA → MoE | FFN → N expertos + router, activa 2/N | Mixtral upcycling |

La columna `params/bytes` de `metrics.csv` ya registra el tamaño por ciclo: el salto se verá en la gráfica de la web.

## Referencias que guían decisiones

- **Karpeles, "The Mathematics of LLMs"** (PDF en Downloads): schedules (cap. 3, aplicado), FFN=memoria KV de hechos (p.800: el aprendizaje por chat escribe en FFN, valida `sesion.bin`), double descent (criterio de expansión), SwiGLU param-counting (p.810, para escalar).
- **Jalapeño inference chip** (zartbot blog): batch decodificación en 1 llamada WASM (aplicado: `ad_generate_n`), layout plano 1 nivel de memoria (aplicado), cuantización heterogénea (pendiente), kernel-loop persistente.
- **NELL (CMU)**: extraer→verificar→acumular = nuestro generate→verify→gate.

## Arquitectura del flujo (punteros)

```
peng-mimo nocturno (dataset C verificado)
  + FineTome-100k (rotante 15MB)  ┐
  + open-perfectblend (rot 10MB)  ├→ train.yml 4×/día
  + sesiones de usuarios (futuro) ┘   gate PPL → commit → Pages
                                        ↓
chat web (WASM, 2MB, instant) ← model.adm ← train.c (cosine/warmup/clip)
  chat aprende en vivo → contrib PR automática → gate → merge
```

## Lecciones fijadas (no repetir)

1. `ft_ingest.c` se guardó como UTF-16 con BOM → gcc falla en CI. Todo fuente nuevo: ASCII/UTF-8 sin BOM.
2. `python3 -c` multilinea dentro de YAML heredoc rompe indentación → siempre ficheros `scripts/*.py` dedicados.
3. Runner ubuntu no trae pyarrow → instalarlo en un step propio (`python3 -m pip install`).
4. Push del bot rechazado si otro run/usuario pusheo durante el entrenamiento → `git pull --rebase` antes del push (aplicado).
5. GitHub rechaza ficheros >100MB: datasets brutos HF nunca al repo; cache Actions o download on-the-fly.
6. Gate de PPL es el árbitro único: publicación = solo mejoras medidas. Lo entrenado en navegador pasa por el mismo gate que lo nocturno.