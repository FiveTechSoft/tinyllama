/*
 * history.js - historico de evolucion del modelo adaptativo
 * Lee adaptive/metrics.csv (publicado en Pages por el workflow) y pinta
 * la curva de PPL + tabla de ciclos.
 */
export async function fetchHistory(base = 'adaptive/') {
  const r = await fetch(base + 'metrics.csv', { cache: 'no-store' });
  if (!r.ok) throw new Error('HTTP ' + r.status);
  const txt = await r.text();
  const rows = [];
  for (const line of txt.split('\n')) {
    if (!line.trim()) continue;
    const m = line.match(/^"?([^",]+)"?,([^,]+),(\d+),([\d.]+),([\d.]+),(\d)/);
    if (m) rows.push({
      date: m[1], model: m[2].trim(), steps: +m[3],
      pplBefore: +m[4], pplAfter: +m[5], published: m[6] === '1',
    });
  }
  return rows;
}

/* sparkline inline SVG de la serie pplAfter */
export function sparkline(ppls, w = 260, h = 42) {
  if (!ppls.length) return '';
  const max = Math.max(...ppls), min = Math.min(...ppls);
  const span = max - min || 1;
  const pts = ppls.map((p, i) => {
    const x = (w - 2) * (ppls.length > 1 ? i / (ppls.length - 1) : 0) + 1;
    const y = 2 + (h - 4) * (1 - (p - min) / span);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return `<svg width="${w}" height="${h}" style="vertical-align:middle">
    <polyline fill="none" stroke="#3fb950" stroke-width="2" points="${pts.join(' ')}"/>
    <circle cx="${pts[pts.length-1].split(',')[0]}" cy="${pts[pts.length-1].split(',')[1]}" r="3" fill="#3fb950"/>
  </svg>`;
}

/* bloque HTML completo: curva + tabla */
export function renderHistory(rows) {
  const pub = rows.filter(r => r.published);
  const line = pub.map(r => r.pplAfter);
  const first = pub[0], last = pub[pub.length - 1];
  const delta = first ? ((last.pplAfter - first.pplAfter) / first.pplAfter * 100) : 0;
  const mejoras = pub.length;
  let html = `<div id="histHead"><b>Ciclos de mejora: ${pub.length}</b> ` +
    sparkline(line) +
    ` <span class="sub">PPL ${(first?.pplAfter ?? 0).toFixed(1)} → ${(last?.pplAfter ?? 0).toFixed(1)} (${delta >= 0 ? '+' : ''}${delta.toFixed(1)}%)</span></div>`;
  html += `<table id="histTable"><tr><th>fecha</th><th>pasos</th><th>PPL antes</th><th>PPL después</th><th>mejora</th></tr>`;
  for (const r of pub.slice(-8).reverse()) {
    const pct = r.pplBefore ? ((r.pplAfter - r.pplBefore) / r.pplBefore * 100) : 0;
    html += `<tr><td>${r.date.slice(5, 16)}</td><td>${r.steps}</td><td>${r.pplBefore.toFixed(1)}</td><td><b>${r.pplAfter.toFixed(1)}</b></td><td style="color:${pct < 0 ? '#3fb950' : '#f85149'}">${pct.toFixed(1)}%</td></tr>`;
  }
  html += '</table>';
  return html;
}