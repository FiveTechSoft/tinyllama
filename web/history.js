/*
 * history.js - historico de evolucion del modelo adaptativo
 * Lee adaptive/metrics.csv y pinta:
 *   - curva de PPL (mejora)
 *   - curva de tamano (params/bytes, si el CSV trae columnas extendidas)
 * Formatos soportados:
 *   viejo: fecha,model,steps,ppl0,ppl1,published
 *   nuevo: fecha,model,steps,ppl0,ppl1,published,params,bytes,dim
 */
export async function fetchHistory(base = 'adaptive/') {
  const r = await fetch(base + 'metrics.csv', { cache: 'no-store' });
  if (!r.ok) throw new Error('HTTP ' + r.status);
  const txt = await r.text();
  const rows = [];
  for (const line of txt.split('\n')) {
    if (!line.trim()) continue;
    const m = line.match(/^"?([^",]+)"?,([^,]+),(\d+),([\d.]+),([\d.]+),(\d)(?:,(\d+),(\d+),(\d+))?/);
    if (m) rows.push({
      date: m[1], model: m[2].trim(), steps: +m[3],
      pplBefore: +m[4], pplAfter: +m[5], published: m[6] === '1',
      params: m[7] ? +m[7] : 0, bytes: m[8] ? +m[8] : 0, dim: m[9] ? +m[9] : 0,
    });
  }
  return rows;
}

/* sparkline inline SVG de una serie */
export function sparkline(vals, color = '#3fb950', w = 260, h = 42) {
  if (!vals || vals.length < 2) return '';
  const max = Math.max(...vals), min = Math.min(...vals);
  const span = max - min || 1;
  const pts = vals.map((p, i) => {
    const x = (w - 2) * (i / (vals.length - 1)) + 1;
    const y = 2 + (h - 4) * (1 - (p - min) / span);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return `<svg width="${w}" height="${h}" style="vertical-align:middle">
    <polyline fill="none" stroke="${color}" stroke-width="2" points="${pts.join(' ')}"/>
    <circle cx="${pts[pts.length-1].split(',')[0]}" cy="${pts[pts.length-1].split(',')[1]}" r="3" fill="${color}"/>
  </svg>`;
}

/* bloque completo: curvas de mejora y tamano + tabla de ciclos */
export function renderHistory(rows) {
  const pub = rows.filter(r => r.published);
  const ppl = pub.map(r => r.pplAfter);
  const hasSize = pub.some(r => r.params > 0);
  const params = pub.map(r => r.params / 1e6);
  const first = pub[0], last = pub[pub.length - 1];
  const delta = first ? ((last.pplAfter - first.pplAfter) / first.pplAfter * 100) : 0;

  let html = `<div><b>Mejora (PPL ↓)</b> ${sparkline(ppl)}` +
    ` <span class="sub">${(first?.pplAfter ?? 0).toFixed(1)} → ${(last?.pplAfter ?? 0).toFixed(1)} (${delta >= 0 ? '+' : ''}${delta.toFixed(1)}%) · ${pub.length} ciclos</span></div>`;
  if (hasSize) {
    html += `<div><b>Tamaño (M pesos ↑)</b> ${sparkline(params, '#2f81f7')}` +
      ` <span class="sub">${params[0].toFixed(2)}M → ${params[params.length-1].toFixed(2)}M · dim=${pub[pub.length-1].dim || '?'}</span></div>`;
  }
  html += `<table id="histTable"><tr><th>fecha</th><th>pasos</th><th>PPL antes</th><th>PPL desp.</th><th>mejora</th>${hasSize ? '<th>pesos</th><th>bytes</th>' : ''}</tr>`;
  for (const r of pub.slice(-10).reverse()) {
    const pct = r.pplBefore ? ((r.pplAfter - r.pplBefore) / r.pplBefore * 100) : 0;
    html += `<tr><td>${r.date.slice(5, 16)}</td><td>${r.steps}</td><td>${r.pplBefore.toFixed(1)}</td><td><b>${r.pplAfter.toFixed(1)}</b></td><td style="color:${pct < 0 ? '#3fb950' : '#f85149'}">${pct.toFixed(1)}%</td>` +
      (hasSize ? `<td>${r.params ? (r.params / 1e6).toFixed(2) + 'M' : '—'}</td><td>${r.bytes ? (r.bytes / 1048576).toFixed(2) + 'MB' : '—'}</td>` : '') +
      `</tr>`;
  }
  html += '</table>';
  return html;
}