// Beta and CAPM regression in the browser. Mirrors src/bc/capm.py; tests compare the two.

// Return minus the risk-free rate over the calendar days since the previous row.
function excess(r, dates, rfAnnual) {
  return r.map((x, i) => {
    if (x === null) return null;
    const days = i === 0 ? 1 : (Date.parse(dates[i] + "T00:00:00Z") - Date.parse(dates[i - 1] + "T00:00:00Z")) / 86400000;
    return x - (Math.pow(1 + rfAnnual, days / 365) - 1);
  });
}

// OLS of y on x with plain and Newey-West (Bartlett, `lags`) standard errors.
function ols(y, x, lags = 5) {
  const n = y.length, mx = x.reduce((a, b) => a + b, 0) / n, my = y.reduce((a, b) => a + b, 0) / n;
  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < n; i++) { sxx += (x[i] - mx) ** 2; sxy += (x[i] - mx) * (y[i] - my); syy += (y[i] - my) ** 2; }
  const beta = sxy / sxx, alpha = my - beta * mx, u = y.map((v, i) => v - alpha - beta * x[i]);
  const ssr = u.reduce((a, b) => a + b * b, 0), s2 = ssr / (n - 2);
  // (XᵀX)⁻¹ for X = [1, x]
  const sx = x.reduce((a, b) => a + b, 0), sx2 = x.reduce((a, b) => a + b * b, 0), det = n * sx2 - sx * sx;
  const inv = [[sx2 / det, -sx / det], [-sx / det, n / det]];
  const S = [[0, 0], [0, 0]], xu = x.map((v, i) => [u[i], v * u[i]]);
  for (let i = 0; i < n; i++) for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) S[a][b] += xu[i][a] * xu[i][b];
  for (let l = 1; l <= lags; l++) {
    const w = 1 - l / (lags + 1);
    for (let i = l; i < n; i++) for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++)
      S[a][b] += w * (xu[i][a] * xu[i - l][b] + xu[i - l][a] * xu[i][b]);
  }
  const mul = (A, B) => [[A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
    [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]];
  const nw = mul(mul(inv, S), inv), k = n / (n - 2);
  return { alpha, beta, se_alpha: Math.sqrt(s2 * inv[0][0]), se_beta: Math.sqrt(s2 * inv[1][1]),
    se_alpha_nw: Math.sqrt(nw[0][0] * k), se_beta_nw: Math.sqrt(nw[1][1] * k), r2: 1 - ssr / syy,
    resid_sd: Math.sqrt(s2), n };
}

// Friday-ending weekly returns (pandas "W-FRI"), first and unfinished last week left out.
function toWeekly(cols, dates) {
  const friday = (iso) => { const d = new Date(iso + "T00:00:00Z"), add = (5 - d.getUTCDay() + 7) % 7;
    d.setUTCDate(d.getUTCDate() + add); return d.toISOString().slice(0, 10); };
  const keys = dates.map(friday), all = [...new Set(keys)], out = {};
  const weeks = all.slice(1).filter((w) => w <= dates[dates.length - 1]);
  for (const [k, v] of Object.entries(cols)) {
    const acc = new Map(all.map((w) => [w, 1]));
    v.forEach((x, t) => { const w = keys[t]; if (acc.get(w) === null) return; acc.set(w, x === null ? null : acc.get(w) * (1 + x)); });
    out[k] = weeks.map((w) => acc.get(w) === null ? null : acc.get(w) - 1);
  }
  return { cols: out, dates: weeks };
}

// Beta over each trailing `w` observations (null until w complete pairs).
function rollingBeta(y, x, w) {
  const out = new Array(y.length).fill(null);
  for (let t = w - 1; t < y.length; t++) {
    const ys = [], xs = [];
    for (let s = t - w + 1; s <= t; s++) { if (y[s] === null || x[s] === null) break; ys.push(y[s]); xs.push(x[s]); }
    if (ys.length < w) continue;
    const my = ys.reduce((a, b) => a + b, 0) / w, mx = xs.reduce((a, b) => a + b, 0) / w;
    let c = 0, v = 0; for (let i = 0; i < w; i++) { c += (xs[i] - mx) * (ys[i] - my); v += (xs[i] - mx) ** 2; }
    out[t] = c / v;
  }
  return out;
}

// Dimson beta: slopes on the market's previous, same and next value, added up (mirrors
// bc/measure.py dimson_beta with one lag). Solves the 4×4 normal equations.
function dimson(y, x) {
  const rows = [];
  for (let t = 1; t < x.length - 1; t++) rows.push([1, x[t - 1], x[t], x[t + 1], y[t]]);
  const A = Array.from({ length: 4 }, () => new Array(5).fill(0));
  for (const r of rows) for (let i = 0; i < 4; i++) { for (let j = 0; j < 4; j++) A[i][j] += r[i] * r[j]; A[i][4] += r[i] * r[4]; }
  for (let c = 0; c < 4; c++) {  // Gauss-Jordan with partial pivoting
    let p = c; for (let i = c + 1; i < 4; i++) if (Math.abs(A[i][c]) > Math.abs(A[p][c])) p = i;
    [A[c], A[p]] = [A[p], A[c]];
    for (let i = 0; i < 4; i++) if (i !== c) { const f = A[i][c] / A[c][c]; for (let j = c; j < 5; j++) A[i][j] -= f * A[c][j]; }
  }
  const coef = A.map((r, i) => r[4] / r[i]);
  return { beta: coef[1] + coef[2] + coef[3], parts: coef.slice(1) };
}

// EWMA beta for day t from days before t (mirrors bc/forecast.py ewma_beta); null until `seed`.
function ewmaBeta(y, x, lam = 0.99, seed = 60) {
  const out = new Array(y.length).fill(null);
  let cxy = 0, vx = 0;
  for (let i = 0; i < seed; i++) { cxy += x[i] * y[i] / seed; vx += x[i] * x[i] / seed; }
  for (let t = seed; t < y.length; t++) {
    if (t > seed) { cxy = lam * cxy + (1 - lam) * x[t - 1] * y[t - 1]; vx = lam * vx + (1 - lam) * x[t - 1] ** 2; }
    out[t] = cxy / vx;
  }
  return out;
}

if (typeof module !== "undefined") module.exports = { excess, ols, toWeekly, rollingBeta, dimson, ewmaBeta };
