/*
 * regresion.js — matemática del proyecto en JavaScript puro (sin DOM).
 * Replica src/analitico.py, derivadas.py, gradiente.py y analisis.py para que
 * app.html recalcule en el navegador. tests/pruebas.py la ejecuta con node
 * para comprobar que da lo mismo que Python (RD-04).
 */
(function (raiz) {
  "use strict";

  function sumas(x, y) {
    const n = x.length;
    let sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < n; i++) {
      sx += x[i]; sy += y[i]; sxy += x[i] * y[i]; sxx += x[i] * x[i]; syy += y[i] * y[i];
    }
    return { n, sx, sy, sxy, sxx, syy, xm: sx / n, ym: sy / n, D: n * sxx - sx * sx };
  }

  // Mismas reglas que datos.validar_datos
  function validar(x, y) {
    if (x.length !== y.length) return "x e y deben tener la misma cantidad de datos.";
    if (x.length < 3) return "Se necesitan al menos 3 observaciones activas.";
    for (let i = 0; i < x.length; i++) {
      if (!Number.isFinite(x[i]) || !Number.isFinite(y[i])) return `Fila ${i + 1}: valor no numérico o no finito.`;
    }
    if (x.every(v => v === x[0])) return "Todos los x son iguales: la recta no es única (det H = 0).";
    return null;
  }

  // Ecuaciones normales: a* = (nΣxy − ΣxΣy)/(nΣx² − (Σx)²), b* = ȳ − a*·x̄
  function ajuste(x, y) {
    const s = sumas(x, y);
    const a = (s.n * s.sxy - s.sx * s.sy) / s.D;
    return { a, b: (s.sy - a * s.sx) / s.n, s };
  }

  function S(x, y, a, b) {
    let t = 0;
    for (let i = 0; i < x.length; i++) { const e = y[i] - a * x[i] - b; t += e * e; }
    return t;
  }

  // S con sumatorias: O(1) por punto, para la malla 3D
  function Ssum(s, a, b) {
    return Math.max(0, s.syy - 2 * a * s.sxy - 2 * b * s.sy + a * a * s.sxx + 2 * a * b * s.sx + s.n * b * b);
  }

  // Variables centradas u = x − x̄, y = a·u + c  (Σu = 0 ⇒ H diagonal)
  function Scentrada(s, a, c) {
    const suu = s.sxx - s.n * s.xm * s.xm, suy = s.sxy - s.xm * s.sy;
    return Math.max(0, s.syy - 2 * a * suy - 2 * c * s.sy + a * a * suu + s.n * c * c);
  }

  function gradiente(x, y, a, b) {
    let ga = 0, gb = 0;
    for (let i = 0; i < x.length; i++) { const e = y[i] - a * x[i] - b; ga -= 2 * x[i] * e; gb -= 2 * e; }
    return [ga, gb];
  }

  const hessiano = s => [[2 * s.sxx, 2 * s.sx], [2 * s.sx, 2 * s.n]];

  // Valores y vectores propios de una matriz simétrica 2×2
  function eig2(H) {
    const p = H[0][0], q = H[0][1], r = H[1][1];
    const det = p * r - q * q, traza = p + r;
    const l2 = traza / 2 + Math.hypot((p - r) / 2, q);
    const l1 = det / l2;                       // evita la cancelación de traza/2 − raíz
    const vec = l => {
      const v = q !== 0 ? [q, l - p] : (Math.abs(l - p) < Math.abs(l - r) ? [1, 0] : [0, 1]);
      const k = Math.hypot(v[0], v[1]);
      return [v[0] / k, v[1] / k];
    };
    return { l1, l2, v1: vec(l1), v2: vec(l2), det, traza, kappa: l2 / l1 };
  }

  // Hessiano por diferencias finitas centradas (igual que analisis.verificar_hessiano_numerico)
  function hessNum(x, y, a, b, h = 1e-3) {
    const f = (p, q) => S(x, y, p, q);
    const Saa = (f(a + h, b) - 2 * f(a, b) + f(a - h, b)) / (h * h);
    const Sbb = (f(a, b + h) - 2 * f(a, b) + f(a, b - h)) / (h * h);
    const Sab = (f(a + h, b + h) - f(a + h, b - h) - f(a - h, b + h) + f(a - h, b - h)) / (4 * h * h);
    return [[Saa, Sab], [Sab, Sbb]];
  }

  const media = v => v.reduce((t, w) => t + w, 0) / v.length;

  function pearson(u, v) {
    const mu = media(u), mv = media(v);
    let suv = 0, suu = 0, svv = 0;
    for (let i = 0; i < u.length; i++) {
      suv += (u[i] - mu) * (v[i] - mv); suu += (u[i] - mu) ** 2; svv += (v[i] - mv) ** 2;
    }
    return suu > 0 && svv > 0 ? suv / Math.sqrt(suu * svv) : 0;
  }

  // §11: residuos con los x ORDENADOS (rachas, regiones y dispersión dependen del orden)
  function residuos(x, y, a, b) {
    const n = x.length;
    const idx = [...x.keys()].sort((i, j) => x[i] - x[j]);
    const xs = idx.map(i => x[i]), ys = idx.map(i => y[i]);
    const e = ys.map((v, i) => v - (a * xs[i] + b));
    const SS = e.reduce((t, v) => t + v * v, 0);
    const s = Math.sqrt(SS / (n - 2));
    const z = e.map(v => (s > 0 ? v / s : 0));
    let rachas = 0, previo = 0;
    for (const v of e) { const g = Math.sign(v); if (g !== 0 && g !== previo) { rachas++; previo = g; } }
    const xm = media(xs), t = Math.floor(n / 3), mitad = Math.floor(n / 2);
    const absMedia = v => media(v.map(Math.abs));
    return {
      idx, xs, ys, e, s, z, rachas,
      media: media(e),
      positivos: e.filter(v => v > 0).length, negativos: e.filter(v => v < 0).length,
      corr: pearson(xs.map(v => (v - xm) ** 2), e),
      regiones: [media(e.slice(0, t)), media(e.slice(t, n - t)), media(e.slice(n - t))],
      disp: [absMedia(e.slice(0, mitad)), absMedia(e.slice(mitad))],
      atipicos: z.map((v, k) => (Math.abs(v) > 2 ? k : -1)).filter(k => k >= 0),
    };
  }

  // §12
  function calidad(x, y, a, b) {
    const n = x.length, ym = media(y);
    let SSres = 0, SStot = 0, abs = 0;
    for (let i = 0; i < n; i++) {
      const e = y[i] - a * x[i] - b; SSres += e * e; SStot += (y[i] - ym) ** 2; abs += Math.abs(e);
    }
    const RMSE = Math.sqrt(SSres / n);
    return { SSres, SStot, R2: SStot > 0 ? 1 - SSres / SStot : NaN, RMSE, MAE: abs / n,
             rmseRel: RMSE / ym, rango: Math.max(...y) - Math.min(...y) };
  }

  // Eliminación de Gauss con pivoteo parcial
  function resolver(M, v) {
    const n = v.length, A = M.map((f, i) => [...f, v[i]]);
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
      [A[c], A[p]] = [A[p], A[c]];
      if (A[c][c] === 0) throw new Error("Sistema singular");
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = A[r][c] / A[c][c];
        for (let k = c; k <= n; k++) A[r][k] -= f * A[c][k];
      }
    }
    return A.map((f, i) => f[n] / f[i]);
  }

  // §13-C: y = p·x² + q·x + r por ecuaciones normales 3×3 + prueba F parcial
  function cuadratico(x, y) {
    const n = x.length, P = k => x.reduce((t, v) => t + v ** k, 0);
    const Py = k => x.reduce((t, v, i) => t + v ** k * y[i], 0);
    const [p, q, r] = resolver([[P(4), P(3), P(2)], [P(3), P(2), P(1)], [P(2), P(1), n]], [Py(2), Py(1), Py(0)]);
    const lin = ajuste(x, y), ym = media(y);
    const Sl = S(x, y, lin.a, lin.b);
    const Sc = x.reduce((t, v, i) => t + (y[i] - p * v * v - q * v - r) ** 2, 0);
    const SStot = y.reduce((t, v) => t + (v - ym) ** 2, 0);
    const difs = x.map(v => Math.abs(p * v * v + q * v + r - (lin.a * v + lin.b)));
    return { p, q, r, Sl, Sc, R2l: 1 - Sl / SStot, R2c: 1 - Sc / SStot,
             RMSEl: Math.sqrt(Sl / n), RMSEc: Math.sqrt(Sc / n),
             F: (Sl - Sc) / (Sc / (n - 3)), maxDif: Math.max(...difs) };
  }

  const ajustable = x => x.length >= 2 && !x.every(v => v === x[0]);

  // §13-B: se elimina cada punto y se reajusta
  function influencia(x, y) {
    const base = ajuste(x, y);
    return x.map((xi, i) => {
      const xx = x.filter((_, j) => j !== i), yy = y.filter((_, j) => j !== i);
      if (!ajustable(xx)) return { i, x: xi, y: y[i], a: NaN, b: NaN, da: NaN, db: NaN, dpred: NaN };
      const f = ajuste(xx, yy);
      const dpred = Math.max(...x.map(v => Math.abs(f.a * v + f.b - (base.a * v + base.b))));
      return { i, x: xi, y: y[i], a: f.a, b: f.b, da: f.a - base.a, db: f.b - base.b, dpred };
    });
  }

  // §13-A: y_i + δ, un punto a la vez
  function sensibilidad(x, y, delta) {
    const base = ajuste(x, y);
    return x.map((_, i) => {
      const yy = y.slice(); yy[i] += delta;
      const f = ajuste(x, yy);
      return { da: f.a - base.a, db: f.b - base.b };
    });
  }

  // Generador pseudoaleatorio con semilla (mulberry32): resultados reproducibles
  function rng(semilla) {
    let t = semilla >>> 0;
    return () => {
      t = (t + 0x6D2B79F5) >>> 0;
      let r = Math.imul(t ^ (t >>> 15), 1 | t);
      r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
      return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
  }

  // §13-D: subconjuntos aleatorios de m = 3..n puntos
  function efectoN(x, y, repeticiones = 200, semilla = 0) {
    const azar = rng(semilla), n = x.length, filas = [];
    const ds = v => Math.sqrt(media(v.map(w => (w - media(v)) ** 2)));
    for (let m = 3; m <= n; m++) {
      const as = [], bs = [];
      for (let k = 0; k < repeticiones; k++) {
        const idx = [...x.keys()];
        for (let i = 0; i < m; i++) {          // Fisher–Yates parcial
          const j = i + Math.floor(azar() * (n - i));
          [idx[i], idx[j]] = [idx[j], idx[i]];
        }
        const xx = idx.slice(0, m).map(i => x[i]), yy = idx.slice(0, m).map(i => y[i]);
        if (!ajustable(xx)) continue;
        const f = ajuste(xx, yy); as.push(f.a); bs.push(f.b);
      }
      filas.push({ m, aMedia: media(as), aDesv: ds(as), bMedia: media(bs), bDesv: ds(bs) });
    }
    return filas;
  }

  // Descenso por gradiente con x centrada (gradiente.py, centrar=True).
  // Devuelve la trayectoria en las variables ORIGINALES (a, b, S).
  function descenso(x, y, a0, b0, maxIter = 20000, tol = 1e-6) {
    const s = sumas(x, y), n = s.n, xm = s.xm;
    const suu = s.sxx - n * xm * xm, suy = s.sxy - xm * s.sy;
    const eta = 1 / (2 * suu + 2 * n);       // 1 / traza(H) en variables centradas
    let a = a0, c = b0 + a0 * xm;
    const tray = [];
    const anotar = () => { const b = c - a * xm; tray.push([a, b, S(x, y, a, b)]); };
    anotar();
    for (let k = 1; k <= maxIter; k++) {
      const ga = 2 * (a * suu - suy), gc = 2 * (n * c - s.sy);
      a -= eta * ga; c -= eta * gc;
      if (k <= 60 || k % 10 === 0) anotar();
      if (Math.hypot(ga, gc) < tol) { anotar(); return { tray, iter: k, convergio: true, eta }; }
    }
    anotar();
    return { tray, iter: maxIter, convergio: false, eta };
  }

  // Rango de la malla alrededor de (a*, b*) siguiendo el valle b = ȳ − a·x̄
  function rangos(aj, Smin, zoom = 1) {
    const s = aj.s, n = s.n;
    const Suu = s.sxx - n * s.xm * s.xm;
    const se = Math.sqrt(Math.max(Smin, 0) / Math.max(n - 2, 1)) / Math.sqrt(Suu);
    const da = zoom * Math.max(10 * se, 0.25 * Math.abs(aj.a), 1e-3 * (1 + Math.abs(aj.a)));
    const db = da * (Math.abs(s.xm) + Math.sqrt(Suu / n));
    const dc = da * Math.sqrt(Suu / n);
    const c = aj.b + aj.a * s.xm;
    return { amin: aj.a - da, amax: aj.a + da, bmin: aj.b - db, bmax: aj.b + db, cmin: c - dc, cmax: c + dc };
  }

  const R = { sumas, validar, ajuste, S, Ssum, Scentrada, gradiente, hessiano, eig2, hessNum,
              residuos, calidad, resolver, cuadratico, influencia, sensibilidad, efectoN,
              descenso, rangos, media };
  if (typeof module !== "undefined" && module.exports) module.exports = R;
  else raiz.R = R;
})(this);
