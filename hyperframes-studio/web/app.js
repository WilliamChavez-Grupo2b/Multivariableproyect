/*
 * app.js — interfaz de app.html: estado, pestañas, vistas enlazadas y gráficos.
 * La matemática vive en regresion.js (objeto global R). Los resultados oficiales
 * de Python llegan en <script id="datos-app"> (generado por src/app_html.py).
 */
(function () {
  "use strict";

  const D = JSON.parse(document.getElementById("datos-app").textContent);
  const $ = id => document.getElementById(id);
  const UX = D.meta.ux, UY = D.meta.uy, UA = `${UY}/${UX}`;
  const EJE_X = `${D.meta.nx} x [${UX}]`, EJE_Y = `${D.meta.ny} y [${UY}]`;
  const CFG = { responsive: true, displaylogo: false, toImageButtonOptions: { format: "png", scale: 2 } };

  // ── Formato ────────────────────────────────────────────────────────────────
  const num = (v, d = 4) => {
    if (v === null || v === undefined || !Number.isFinite(v)) return "—";
    if (v !== 0 && (Math.abs(v) >= 1e7 || Math.abs(v) < 1e-4)) return v.toExponential(3);
    return Number.isInteger(v) ? String(v) : v.toFixed(d);
  };
  const exp = v => (v === null || !Number.isFinite(v) ? "—" : v.toExponential(2));
  const conSigno = v => (v < 0 ? `− ${num(-v)}` : `+ ${num(v)}`);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const lin = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const color = n => getComputedStyle(document.documentElement).getPropertyValue("--" + n).trim();
  const guardar = (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* sin almacenamiento */ } };
  const leer = k => { try { return localStorage.getItem(k); } catch (e) { return null; } };

  // ── Estado ─────────────────────────────────────────────────────────────────
  const st = {
    filas: D.x.map((x, i) => ({ x, y: D.y[i], activo: true })),
    a: NaN, b: NaN,
    manual: { a: D.manual.a, b: D.manual.b, S: D.manual.S },
    log: true, centrado: false, zoom: 1, delta: 1,
    tab: "datos",
  };
  const firmaOriginal = JSON.stringify(D.x.map((x, i) => [x, D.y[i]]));
  const firma = () => JSON.stringify(st.filas.filter(f => f.activo).map(f => [f.x, f.y]));
  let M = null;   // resultados del recálculo con los datos actuales

  function recalcular() {
    const act = st.filas.filter(f => f.activo);
    const x = act.map(f => f.x), y = act.map(f => f.y);
    const err = R.validar(x, y);
    $("error").hidden = !err;
    $("error").textContent = err ? "Datos no válidos: " + err : "";
    if (err) { M = null; return; }
    const aj = R.ajuste(x, y), Smin = R.S(x, y, aj.a, aj.b), H = R.hessiano(aj.s);
    const rg = R.rangos(aj, Smin, st.zoom);
    M = {
      act, x, y, aj, s: aj.s, Smin, H, eig: R.eig2(H), rg,
      Hn: R.hessNum(x, y, aj.a, aj.b), Hn0: R.hessNum(x, y, 0, 0),
      res: R.residuos(x, y, aj.a, aj.b), cal: R.calidad(x, y, aj.a, aj.b),
      cuad: x.length >= 4 && new Set(x).size >= 3 ? R.cuadratico(x, y) : null,
      infl: R.influencia(x, y), efn: R.efectoN(x, y, 200, 0),
      gd: R.descenso(x, y, rg.amin + 0.15 * (rg.amax - rg.amin), rg.bmax - 0.15 * (rg.bmax - rg.bmin)),
      modificado: firma() !== firmaOriginal,
      mallas: {},
    };
    if (!Number.isFinite(st.a) || !Number.isFinite(st.b)) rectaPrueba();
    st.a = clamp(st.a, rg.amin, rg.amax);
    st.b = clamp(st.b, rg.bmin, rg.bmax);
    configurarControles();
  }

  function rectaPrueba() {
    st.a = M.aj.a + 0.35 * (M.rg.amax - M.aj.a);
    st.b = M.aj.b;
  }

  // Malla de S (60×60). Original: (a, b). Centrada: (a, c) con c = b + a·x̄.
  function malla(centrado, log) {
    const k = `${centrado}|${log}`;
    if (M.mallas[k]) return M.mallas[k];
    const rg = M.rg, s = M.s, N = 60;
    const f = centrado ? (a, c) => R.Scentrada(s, a, c) : (a, b) => R.Ssum(s, a, b);
    const A = lin(rg.amin, rg.amax, N);
    const B = centrado ? lin(rg.cmin, rg.cmax, N) : lin(rg.bmin, rg.bmax, N);
    const S = B.map(b => A.map(a => f(a, b)));
    const t = v => (log ? Math.log10(Math.max(v, 1e-12)) : v);
    const Z = S.map(fila => fila.map(t));
    const plano = Z.flat();
    return (M.mallas[k] = { A, B, S, Z, f, t, zmin: Math.min(...plano), zmax: Math.max(...plano) });
  }

  // ── Utilidades de gráficos ─────────────────────────────────────────────────
  function lay(titulo, xt, yt, extra) {
    const eje = t => ({ title: { text: t }, gridcolor: color("grid"), zerolinecolor: color("line"),
                        linecolor: color("line"), automargin: true });
    return Object.assign({
      title: { text: titulo, font: { size: 14 } },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: color("fg"), family: "system-ui, sans-serif", size: 12 },
      margin: { l: 60, r: 20, t: 44, b: 50 },
      xaxis: eje(xt), yaxis: eje(yt),
      legend: { orientation: "h", y: -0.2 },
      hoverlabel: { namelength: -1 },
    }, extra || {});
  }
  const dibujar = (id, datos, layout) => Plotly.react($(id), datos, layout, CFG);

  const kpis = (id, lista) => {
    $(id).innerHTML = lista.map(([t, v]) => `<div class="kpi"><b>${v}</b><span>${t}</span></div>`).join("");
  };

  function descargar(nombre, filas) {
    const celda = v => (typeof v === "number" ? (Number.isFinite(v) ? String(v) : "")
                                               : `"${String(v).replace(/"/g, '""')}"`);
    const txt = filas.map(f => f.map(celda).join(",")).join("\n");
    const url = URL.createObjectURL(new Blob(["﻿" + txt], { type: "text/csv;charset=utf-8" }));
    const enlace = Object.assign(document.createElement("a"), { href: url, download: nombre });
    document.body.append(enlace); enlace.click(); enlace.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // Curvas de nivel de log₁₀ S en (a, b) + óptimo; la usan las vistas 3 y 6
  function contorno(extra) {
    const m = malla(false, true);
    return [{
      type: "contour", x: m.A, y: m.B, z: m.Z, customdata: m.S, colorscale: "Viridis", ncontours: 24,
      contours: { coloring: "heatmap", showlines: true }, line: { width: 0.6, color: "rgba(255,255,255,.35)" },
      colorbar: { title: { text: "log₁₀ S" }, thickness: 12 }, name: "log₁₀ S",
      hovertemplate: "a = %{x:.4f}<br>b = %{y:.3f}<br>S = %{customdata:.5g}<extra></extra>",
    }, {
      type: "scatter", x: [M.aj.a], y: [M.aj.b], mode: "markers", name: "(a*, b*)",
      marker: { symbol: "star", size: 16, color: color("opt"), line: { color: "#fff", width: 1 } },
      hovertemplate: `a* = ${num(M.aj.a)}<br>b* = ${num(M.aj.b)}<br>S* = ${num(M.Smin)}<extra></extra>`,
    }, ...extra];
  }

  // ── Vista 1 · Datos ────────────────────────────────────────────────────────
  function construirTabla() {
    $("tablaDatos").innerHTML =
      `<thead><tr><th>i</th><th>x [${esc(UX)}]</th><th>y [${esc(UY)}]</th><th>activo</th><th></th></tr></thead><tbody>` +
      st.filas.map((f, i) => `<tr><td>${i + 1}</td>
        <td><input type="number" step="any" data-i="${i}" data-k="x" value="${Number.isFinite(f.x) ? f.x : ""}" aria-label="x de la fila ${i + 1}"></td>
        <td><input type="number" step="any" data-i="${i}" data-k="y" value="${Number.isFinite(f.y) ? f.y : ""}" aria-label="y de la fila ${i + 1}"></td>
        <td><input type="checkbox" data-i="${i}" data-k="activo" ${f.activo ? "checked" : ""} aria-label="Fila ${i + 1} activa"></td>
        <td><button class="btn" type="button" data-borrar="${i}" aria-label="Eliminar fila ${i + 1}">✕</button></td></tr>`).join("") +
      "</tbody>";
  }

  function vDatos() {
    if (!M) return;
    const ina = st.filas.filter(f => !f.activo && Number.isFinite(f.x) && Number.isFinite(f.y));
    const xr = [Math.min(...M.x), Math.max(...M.x)];
    dibujar("gDispersion", [
      { x: M.x, y: M.y, mode: "markers", type: "scatter", name: "Datos", marker: { color: color("dato"), size: 8 } },
      { x: ina.map(f => f.x), y: ina.map(f => f.y), mode: "markers", type: "scatter", name: "Excluidos",
        marker: { color: color("dato"), size: 9, symbol: "circle-open" } },
      { x: xr, y: xr.map(v => M.aj.a * v + M.aj.b), mode: "lines", type: "scatter",
        name: `ŷ = ${num(M.aj.a)}·x ${conSigno(M.aj.b)}`, line: { color: color("opt"), width: 2 } },
    ], lay("Diagrama de dispersión y recta óptima", EJE_X, EJE_Y));
    const s = M.s;
    kpis("kpisDatos", [["n", s.n], ["Σx", num(s.sx)], ["Σy", num(s.sy)], ["Σxy", num(s.sxy)],
                       ["Σx²", num(s.sxx)], ["Σy²", num(s.syy)], ["x̄", num(s.xm)], ["ȳ", num(s.ym)]]);
  }

  function cargarCSV(texto) {
    const lineas = texto.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
    if (!lineas.length) throw new Error("El archivo está vacío.");
    const sep = lineas[0].includes(";") ? ";" : ",";
    const celdas = l => l.split(sep).map(c => c.trim().replace(/^"|"$/g, ""));
    let ix = 0, iy = 1, inicio = 0;
    const cab = celdas(lineas[0]).map(c => c.toLowerCase());
    if (cab.some(c => c !== "" && !Number.isFinite(Number(c)))) {
      inicio = 1;
      if (cab.includes("x") && cab.includes("y")) { ix = cab.indexOf("x"); iy = cab.indexOf("y"); }
    }
    const filas = lineas.slice(inicio).map((l, k) => {
      const c = celdas(l), x = Number(c[ix]), y = Number(c[iy]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) throw new Error(`Línea ${k + inicio + 1}: valor no numérico → ${l}`);
      return { x, y, activo: true };
    });
    st.filas = filas;
  }

  // ── Vista 2 · Analítico ────────────────────────────────────────────────────
  function vAnalitico() {
    const s = M.s, a = M.aj.a, b = M.aj.b, g = R.gradiente(M.x, M.y, a, b);
    const numA = s.n * s.sxy - s.sx * s.sy;
    $("desarrollo").innerHTML = `
      <h3>1 · Función de error (n = ${s.n})</h3>
      <div class="paso">S(a, b) = Σ (yᵢ − a·xᵢ − b)²</div>
      <h3>2 · Derivadas parciales (regla de la cadena)</h3>
      <div class="paso">∂S/∂a = Σ 2(yᵢ − a·xᵢ − b)·(−xᵢ) = −2 Σ xᵢ(yᵢ − a·xᵢ − b)</div>
      <div class="paso">∂S/∂b = Σ 2(yᵢ − a·xᵢ − b)·(−1) = −2 Σ (yᵢ − a·xᵢ − b)</div>
      <h3>3 · Punto crítico ∇S = 0 (ecuaciones normales)</h3>
      <div class="paso">a·Σx² + b·Σx = Σxy &nbsp;→&nbsp; ${num(s.sxx)}·a + ${num(s.sx)}·b = ${num(s.sxy)}</div>
      <div class="paso">a·Σx + n·b = Σy &nbsp;→&nbsp; ${num(s.sx)}·a + ${s.n}·b = ${num(s.sy)}</div>
      <h3>4 · Solución del sistema</h3>
      <div class="paso">a* = (nΣxy − ΣxΣy) / (nΣx² − (Σx)²) = (${s.n}·${num(s.sxy)} − ${num(s.sx)}·${num(s.sy)}) / (${s.n}·${num(s.sxx)} − ${num(s.sx)}²) = ${num(numA)} / ${num(s.D)} = <b>${num(a, 6)}</b></div>
      <div class="paso">b* = ȳ − a*·x̄ = ${num(s.ym)} − ${num(a)}·${num(s.xm)} = <b>${num(b, 6)}</b></div>
      <div class="paso">S(a*, b*) = <b>${num(M.Smin, 6)}</b> ${esc(UY)}² &nbsp;·&nbsp; ∇S(a*, b*) = (${exp(g[0])}, ${exp(g[1])}) ≈ 0</div>
      <div class="paso">Recta de ajuste: <b>ŷ = ${num(a)}·x ${conSigno(b)}</b></div>`;
    const H = M.H;
    $("hessResumen").innerHTML = `
      <h3>5 · Hessiano (§7.4)</h3>
      <div class="paso">H = 2·[[Σx², Σx], [Σx, n]] = [[${num(H[0][0])}, ${num(H[0][1])}], [${num(H[1][0])}, ${num(H[1][1])}]]</div>
      <div class="paso">∂²S/∂a² = 2Σx² = ${num(H[0][0])} ${H[0][0] > 0 ? '<span class="si">&gt; 0 ✓</span>' : '<span class="no">≤ 0 ✗</span>'}</div>
      <div class="paso">det H = 4(nΣx² − (Σx)²) = ${num(M.eig.det)} ${M.eig.det > 0 ? '<span class="si">&gt; 0 ✓</span>' : '<span class="no">≤ 0 ✗</span>'}</div>
      <div class="paso">⇒ H es definida positiva y constante: (a*, b*) es el <b>mínimo global</b> de S.</div>`;
    $("manA").value = st.manual.a; $("manB").value = st.manual.b; $("manS").value = st.manual.S;
    textoManual();
  }

  function textoManual() {
    if (!M) return;
    const d = (m, c) => (Number.isFinite(m) ? exp(Math.abs(m - c)) : "—");
    $("difManual").innerHTML = `Diferencia con el código: |Δa| = ${d(st.manual.a, M.aj.a)} · |Δb| = ${d(st.manual.b, M.aj.b)} · |ΔS| = ${d(st.manual.S, M.Smin)}`;
  }

  // ── Controles (a, b) e indicador ───────────────────────────────────────────
  function configurarControles() {
    const rg = M.rg;
    for (const [id, min, max, v] of [["rangoA", rg.amin, rg.amax, st.a], ["rangoB", rg.bmin, rg.bmax, st.b]]) {
      const el = $(id);
      el.min = min; el.max = max; el.step = (max - min) / 1000; el.value = v;
    }
    $("numA").value = +st.a.toFixed(6); $("numB").value = +st.b.toFixed(4);
    $("rangoA").setAttribute("aria-valuetext", `a = ${num(st.a)}`);
    $("rangoB").setAttribute("aria-valuetext", `b = ${num(st.b)}`);
  }

  function indicador() {
    const S = R.S(M.x, M.y, st.a, st.b), extra = M.Smin > 0 ? (S / M.Smin - 1) * 100 : 0;
    $("indicador").innerHTML = `S(a, b) = <b>${num(S)}</b> ${esc(UY)}² &nbsp;|&nbsp; S* = ${num(M.Smin)} &nbsp;|&nbsp; ${M.Smin > 0 ? `+${num(extra, 1)} % sobre el mínimo` : ""}`;
    // medidor: 100 % en el mínimo, baja en escala logarítmica
    const r = Math.log10(Math.max(S, 1e-12) / Math.max(M.Smin, 1e-12));
    $("medidor").style.width = `${clamp(100 - r * 25, 2, 100)}%`;
  }

  function moverAB(a, b) {
    if (!M) return;
    st.a = clamp(a, M.rg.amin, M.rg.amax);
    st.b = clamp(b, M.rg.bmin, M.rg.bmax);
    configurarControles();
    render();
  }

  // ── Vista 3 · Explorador ───────────────────────────────────────────────────
  let clicContorno = false;
  function vExplorador() {
    const { x, y } = M, a = st.a, b = st.b, xr = [Math.min(...x), Math.max(...x)];
    const segX = [], segY = [];
    x.forEach((xi, i) => { segX.push(xi, xi, null); segY.push(y[i], a * xi + b, null); });
    dibujar("gRecta", [
      { x: segX, y: segY, mode: "lines", type: "scatter", name: "Residuos eᵢ", line: { color: color("usr"), width: 1, dash: "dot" }, hoverinfo: "skip" },
      { x, y, mode: "markers", type: "scatter", name: "Datos", marker: { color: color("dato"), size: 8 } },
      { x: xr, y: xr.map(v => M.aj.a * v + M.aj.b), mode: "lines", type: "scatter", name: "Recta óptima (a*, b*)", line: { color: color("opt"), width: 2 } },
      { x: xr, y: xr.map(v => a * v + b), mode: "lines", type: "scatter", name: "Recta (a, b)", line: { color: color("usr"), width: 2.5 } },
    ], lay(`Recta ŷ = ${num(a)}·x ${conSigno(b)}  ·  S = ${num(R.S(x, y, a, b))}`, EJE_X, EJE_Y));

    const gd = M.gd.tray;
    dibujar("gContorno", contorno([
      { type: "scatter", x: gd.map(p => p[0]), y: gd.map(p => p[1]), mode: "lines", name: "Descenso por gradiente",
        line: { color: "#fff", width: 1.5, dash: "dot" }, visible: "legendonly", hoverinfo: "skip" },
      { type: "scatter", x: [a], y: [b], mode: "markers", name: "(a, b) actual",
        marker: { size: 13, color: color("usr"), line: { color: "#fff", width: 2 } } },
    ]), lay("Curvas de nivel de S(a, b)", `a [${UA}]`, `b [${UY}]`));
    if (!clicContorno) {
      $("gContorno").on("plotly_click", e => { const p = e.points[0]; if (p) moverAB(p.x, p.y); });
      clicContorno = true;
    }
  }

  // ── Vista 4 · Superficie 3D ────────────────────────────────────────────────
  const CAMARAS = {
    iso: { eye: { x: 1.6, y: -1.6, z: 0.9 }, up: { x: 0, y: 0, z: 1 } },
    arriba: { eye: { x: 0, y: 0, z: 2.4 }, up: { x: 0, y: 1, z: 0 } },
    a: { eye: { x: 0, y: -2.4, z: 0.25 }, up: { x: 0, y: 0, z: 1 } },
    b: { eye: { x: 2.4, y: 0, z: 0.25 }, up: { x: 0, y: 0, z: 1 } },
  };

  function v3D() {
    const c = st.centrado, m = malla(c, st.log), xm = M.s.xm;
    const yv = (a, b) => (c ? b + a * xm : b);                 // b → c en variables centradas
    const A = m.A, B = m.B, a0 = st.a, y0 = yv(st.a, st.b);
    const z = (a, w) => m.t(m.f(a, w));
    const nomY = c ? "c" : "b";
    const ejeZ = st.log ? "log₁₀ S" : `S [${UY}²]`;
    const linea = (pts, name, col, extra) => Object.assign({
      type: "scatter3d", mode: "lines", name, x: pts.map(p => p[0]), y: pts.map(p => p[1]),
      z: pts.map(p => z(p[0], p[1])), line: { color: col, width: 6 }, hoverinfo: "skip",
    }, extra);
    const uni = col => [[0, col], [1, col]];
    const plano = (xs, ys, name, col) => ({
      type: "surface", name, x: xs, y: ys, z: [[m.zmin, m.zmin], [m.zmax, m.zmax]], opacity: 0.22,
      colorscale: uni(col), showscale: false, showlegend: true, hoverinfo: "skip", visible: ver,
    });
    const dentro = p => p[0] >= A[0] && p[0] <= A[A.length - 1] && p[1] >= B[0] && p[1] <= B[B.length - 1];
    const gd = M.gd.tray.map(p => [p[0], yv(p[0], p[1])]).filter(dentro);
    const ver = dentro([a0, y0]);                              // (a, b) actual dentro de la malla

    dibujar("g3d", [
      { type: "surface", name: "S(a, b)", x: A, y: B, z: m.Z, customdata: m.S, colorscale: "Viridis",
        opacity: 0.93, showlegend: true,
        contours: { z: { show: true, usecolormap: true, project: { z: true },
                         start: m.zmin, end: m.zmax, size: (m.zmax - m.zmin) / 16 } },
        colorbar: { title: { text: ejeZ }, len: 0.6, thickness: 12 },
        hovertemplate: `a = %{x:.4f}<br>${nomY} = %{y:.4f}<br>S = %{customdata:.5g}<extra></extra>` },
      plano([[a0, a0], [a0, a0]], [[B[0], B[B.length - 1]], [B[0], B[B.length - 1]]], `Plano a = ${num(a0)}`, color("usr")),
      plano([[A[0], A[A.length - 1]], [A[0], A[A.length - 1]]], [[y0, y0], [y0, y0]], `Plano ${nomY} = ${num(y0)}`, color("neg")),
      linea(B.map(w => [a0, w]), `Traza a = ${num(a0)}`, color("usr"), { visible: ver }),
      linea(A.map(a => [a, y0]), `Traza ${nomY} = ${num(y0)}`, color("neg"), { visible: ver }),
      linea(A.map(a => [a, c ? M.s.ym : M.s.ym - a * xm]), c ? "Valle c = ȳ" : "Valle b = ȳ − a·x̄",
            "#ffffff", { line: { color: "#ffffff", width: 4, dash: "dash" }, visible: "legendonly" }),
      Object.assign(linea(gd, `Descenso por gradiente (${M.gd.iter} it.)`, color("pos")),
                    { mode: "lines+markers", marker: { size: 2, color: color("pos") }, visible: "legendonly" }),
      { type: "scatter3d", mode: "markers+text", name: "Mínimo (a*, b*)", x: [M.aj.a], y: [yv(M.aj.a, M.aj.b)],
        z: [m.t(M.Smin)], text: ["(a*, b*)"], textposition: "top center",
        marker: { size: 7, color: color("opt"), symbol: "diamond" },
        hovertemplate: `a* = ${num(M.aj.a)}<br>${nomY}* = ${num(yv(M.aj.a, M.aj.b))}<br>S* = ${num(M.Smin)}<extra></extra>` },
      { type: "scatter3d", mode: "markers", name: "(a, b) actual", x: [a0], y: [y0], z: [z(a0, y0)],
        marker: { size: 6, color: color("usr"), line: { color: "#fff", width: 1 } }, visible: ver,
        hovertemplate: `a = ${num(a0)}<br>${nomY} = ${num(y0)}<br>S = ${num(m.f(a0, y0))}<extra></extra>` },
    ], lay("", "", "", {
      margin: { l: 0, r: 0, t: 10, b: 0 },
      uirevision: c ? "centrado" : "original",
      legend: { orientation: "h", y: 0 },
      scene: {
        xaxis: { title: { text: `a [${UA}]` }, gridcolor: color("grid"), showbackground: false, range: [A[0], A[A.length - 1]] },
        yaxis: { title: { text: c ? `c = b + a·x̄ [${UY}]` : `b [${UY}]` }, gridcolor: color("grid"), showbackground: false, range: [B[0], B[B.length - 1]] },
        zaxis: { title: { text: ejeZ }, gridcolor: color("grid"), showbackground: false },
        camera: CAMARAS.iso, aspectmode: "cube",
      },
    }));
    $("nota3d").textContent = (ver ? "" : `El punto (a, b) actual (${nomY} = ${num(y0)}) queda fuera del rango visible: usa "Ir al óptimo" o el zoom. `) + (c
      ? "Con x centrada el Hessiano es diagonal: el valle queda alineado con los ejes y la superficie es un tazón. Activa capas desde la leyenda."
      : `Valle largo y estrecho: κ(H) = ${num(M.eig.kappa, 0)}. Activa u oculta capas desde la leyenda (planos, trazas, valle, descenso por gradiente).`);
  }

  // ── Vista 5 · Trazas ───────────────────────────────────────────────────────
  function vTrazas() {
    const s = M.s, rg = M.rg, A = lin(rg.amin, rg.amax, 200), B = lin(rg.bmin, rg.bmax, 200);
    const f = (a, b) => R.Ssum(s, a, b);
    const aV = (s.sxy - st.b * s.sx) / s.sxx;           // vértice de S(a; b₀)
    const bV = (s.sy - st.a * s.sx) / s.n;              // vértice de S(a₀; b)
    dibujar("gTrazaA", [
      { x: A, y: A.map(a => f(a, M.aj.b)), mode: "lines", type: "scatter", name: `b = b* (pasa por el mínimo)`, line: { color: color("opt"), dash: "dash" } },
      { x: A, y: A.map(a => f(a, st.b)), mode: "lines", type: "scatter", name: `b₀ = ${num(st.b)}`, line: { color: color("usr"), width: 2.5 } },
      { x: [aV], y: [f(aV, st.b)], mode: "markers", type: "scatter", name: "Vértice", marker: { color: color("usr"), size: 10, symbol: "diamond" } },
      { x: [M.aj.a], y: [M.Smin], mode: "markers", type: "scatter", name: "(a*, S*)", marker: { color: color("opt"), size: 12, symbol: "star" } },
    ], lay("Traza con b = b₀ fijo: parábola en a", `a [${UA}]`, `S [${UY}²]`));
    dibujar("gTrazaB", [
      { x: B, y: B.map(b => f(M.aj.a, b)), mode: "lines", type: "scatter", name: `a = a* (pasa por el mínimo)`, line: { color: color("opt"), dash: "dash" } },
      { x: B, y: B.map(b => f(st.a, b)), mode: "lines", type: "scatter", name: `a₀ = ${num(st.a)}`, line: { color: color("neg"), width: 2.5 } },
      { x: [bV], y: [f(st.a, bV)], mode: "markers", type: "scatter", name: "Vértice", marker: { color: color("neg"), size: 10, symbol: "diamond" } },
      { x: [M.aj.b], y: [M.Smin], mode: "markers", type: "scatter", name: "(b*, S*)", marker: { color: color("opt"), size: 12, symbol: "star" } },
    ], lay("Traza con a = a₀ fijo: parábola en b", `b [${UY}]`, `S [${UY}²]`));
    $("notaTrazaA").textContent = `Vértice en a = (Σxy − b₀Σx)/Σx² = ${num(aV)}. Solo cuando b₀ = b* el vértice es a* = ${num(M.aj.a)}. Curvatura ∂²S/∂a² = 2Σx² = ${num(M.H[0][0])} (muy cerrada).`;
    $("notaTrazaB").textContent = `Vértice en b = (Σy − a₀Σx)/n = ${num(bV)}. Solo cuando a₀ = a* el vértice es b* = ${num(M.aj.b)}. Curvatura ∂²S/∂b² = 2n = ${num(M.H[1][1])} (muy abierta).`;
  }

  // ── Vista 6 · Hessiano ─────────────────────────────────────────────────────
  function vHessiano() {
    const H = M.H, Hn = M.Hn, H0 = M.Hn0, e = M.eig;
    const errRel = Math.max(...[0, 1].flatMap(i => [0, 1].map(j => Math.abs(Hn[i][j] - H[i][j])))) / Math.abs(H[0][0]);
    const fila = (n, f, i, j) => `<tr><td>${n}</td><td>${f}</td><td>${num(H[i][j])}</td><td>${num(Hn[i][j])}</td><td>${num(H0[i][j])}</td></tr>`;
    const ok = c => (c ? '<span class="si">✓</span>' : '<span class="no">✗</span>');
    $("hessTabla").innerHTML = `
      <h3>Derivadas de segundo orden</h3>
      <div class="tabla-scroll"><table>
        <thead><tr><th>Entrada</th><th>Fórmula</th><th>Analítico</th><th>Dif. finitas en (a*, b*)</th><th>Dif. finitas en (0, 0)</th></tr></thead>
        <tbody>${fila("∂²S/∂a²", "2Σx²", 0, 0)}${fila("∂²S/∂a∂b", "2Σx", 0, 1)}${fila("∂²S/∂b²", "2n", 1, 1)}</tbody>
      </table></div>
      <p class="nota">Error relativo de las diferencias finitas: ${exp(errRel)}. H es igual en (a*, b*) y en (0, 0): no depende de a ni de b.</p>
      <div class="kpis">
        <div class="kpi"><b>${num(e.det)}</b><span>det H = 4(nΣx² − (Σx)²)</span></div>
        <div class="kpi"><b>${num(e.traza)}</b><span>traza H</span></div>
        <div class="kpi"><b>${num(e.l1)}</b><span>λ₁ (dirección del valle)</span></div>
        <div class="kpi"><b>${num(e.l2)}</b><span>λ₂</span></div>
        <div class="kpi"><b>${num(e.kappa, 0)}</b><span>κ = λ₂ / λ₁</span></div>
      </div>
      <h3 style="margin-top:12px">Criterio del Hessiano</h3>
      <p>${ok(H[0][0] > 0)} ∂²S/∂a² &gt; 0 &nbsp; ${ok(e.det > 0)} det H &gt; 0 &nbsp; ${ok(e.l1 > 0 && e.l2 > 0)} λ₁, λ₂ &gt; 0 &nbsp; ${ok(Math.abs(H[0][1] - H[1][0]) === 0)} simétrica</p>
      <p>${e.det > 0 && H[0][0] > 0
        ? "H es <b>definida positiva</b> en todo el plano: S es convexa y (a*, b*) es su <b>mínimo global</b> único."
        : '<span class="no">No se puede confirmar el mínimo.</span>'}
      κ grande significa elipses muy alargadas: S cambia mucho al mover a y muy poco al moverse a lo largo del valle.</p>`;
    const rg = M.rg, da = rg.amax - M.aj.a, db = rg.bmax - M.aj.b;
    const flecha = (v, l, nombre, col) => {
      const t = 0.8 * Math.min(Math.abs(v[0]) > 0 ? da / Math.abs(v[0]) : Infinity, Math.abs(v[1]) > 0 ? db / Math.abs(v[1]) : Infinity);
      return { type: "scatter", mode: "lines+markers", name: `${nombre} (λ = ${num(l)})`,
               x: [M.aj.a - t * v[0], M.aj.a + t * v[0]], y: [M.aj.b - t * v[1], M.aj.b + t * v[1]],
               line: { color: col, width: 3 }, marker: { size: 7, symbol: "triangle-up" } };
    };
    dibujar("gEigen", contorno([flecha(e.v1, e.l1, "v₁ valle", "#ffffff"), flecha(e.v2, e.l2, "v₂", color("neg"))]),
            lay("Vectores propios de H sobre las curvas de nivel", `a [${UA}]`, `b [${UY}]`));
  }

  // ── Vista 7 · Residuos ─────────────────────────────────────────────────────
  function vResiduos() {
    const r = M.res, q = M.cal, n = r.e.length;
    kpis("kpisResiduos", [
      ["R²", num(q.R2, 5)], [`RMSE [${UY}]`, num(q.RMSE)], [`MAE [${UY}]`, num(q.MAE)],
      ["RMSE / ȳ", `${num(100 * q.rmseRel, 2)} %`], ["Media de eᵢ", exp(r.media)],
      ["Rachas de signo", `${r.rachas} de ${n}`], ["corr(e, (x − x̄)²)", num(r.corr, 3)], ["Atípicos |z| > 2", r.atipicos.length],
    ]);
    const pos = r.e.map((v, k) => (v >= 0 ? k : -1)).filter(k => k >= 0), neg = r.e.map((v, k) => (v < 0 ? k : -1)).filter(k => k >= 0);
    const tr = (ks, name, col, sym) => ({ type: "scatter", mode: "markers", name, x: ks.map(k => r.xs[k]), y: ks.map(k => r.e[k]),
      marker: { color: col, size: 10, symbol: sym }, hovertemplate: "x = %{x}<br>e = %{y:.4f}<extra></extra>" });
    const xr = [r.xs[0], r.xs[n - 1]];
    dibujar("gResiduos", [
      { type: "scatter", mode: "lines", x: r.xs, y: r.e, name: "Tendencia", line: { color: color("muted"), width: 1 }, hoverinfo: "skip" },
      tr(pos, "eᵢ ≥ 0 (dato sobre la recta)", color("pos"), "triangle-up"),
      tr(neg, "eᵢ < 0 (dato bajo la recta)", color("neg"), "triangle-down"),
      { type: "scatter", mode: "markers", name: "Atípico (|z| > 2)", x: r.atipicos.map(k => r.xs[k]), y: r.atipicos.map(k => r.e[k]),
        marker: { size: 20, color: "rgba(0,0,0,0)", line: { color: color("opt"), width: 2 } } },
      { type: "scatter", mode: "lines", x: xr, y: [2 * r.s, 2 * r.s], name: "± 2s", line: { color: color("muted"), dash: "dash", width: 1 } },
      { type: "scatter", mode: "lines", x: xr, y: [-2 * r.s, -2 * r.s], showlegend: false, line: { color: color("muted"), dash: "dash", width: 1 } },
    ], lay("Residuos eᵢ = yᵢ − (a*xᵢ + b*) frente a xᵢ", EJE_X, `e [${UY}]`,
           { shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: color("fg"), width: 1 } }] }));
    dibujar("gHistograma", [{ type: "histogram", x: r.e, nbinsx: Math.ceil(1 + Math.log2(n)), name: "Residuos",
      marker: { color: color("acento"), line: { color: color("panel"), width: 1 } } }],
      lay(`Histograma de residuos (Sturges: ${Math.ceil(1 + Math.log2(n))} clases)`, `e [${UY}]`, "Frecuencia", { bargap: 0.05 }));

    const [izq, cen, der] = r.regiones, sg = v => (v > 0 ? "+" : "−");
    const sistematica = Math.sign(izq) === Math.sign(der) && Math.sign(cen) !== Math.sign(izq);
    const ratio = r.disp[1] / r.disp[0];
    $("interpretacion").innerHTML = `<h3>Interpretación automática (§11–12)</h3><ul>
      <li><b>Centrados en cero:</b> media = ${exp(r.media)}. Es exactamente 0 porque ∂S/∂b = −2Σeᵢ = 0 en el óptimo.</li>
      <li><b>Patrón:</b> ${Math.abs(r.corr) > 0.5
        ? `corr(e, (x − x̄)²) = ${num(r.corr, 3)} y solo ${r.rachas} rachas de signo: hay un <b>patrón curvo</b> (forma de ${r.corr > 0 ? "U" : "∩"}).`
        : `corr(e, (x − x̄)²) = ${num(r.corr, 3)} y ${r.rachas} rachas: no se ve un patrón claro.`}</li>
      <li><b>Regiones:</b> residuo medio izquierda ${sg(izq)}${num(Math.abs(izq), 3)}, centro ${sg(cen)}${num(Math.abs(cen), 3)}, derecha ${sg(der)}${num(Math.abs(der), 3)} ${esc(UY)}.
        ${sistematica ? "Los extremos y el centro tienen signos opuestos: <b>desviación sistemática</b>; la relación real es levemente curva." : "No hay un cambio de signo sistemático por regiones."}</li>
      <li><b>Dispersión:</b> |e| medio ${num(r.disp[0], 3)} en la primera mitad y ${num(r.disp[1], 3)} en la segunda
        (${ratio > 1.5 ? "aumenta con x" : ratio < 1 / 1.5 ? "disminuye con x" : "similar"}).</li>
      <li><b>Atípicos:</b> ${r.atipicos.length ? r.atipicos.map(k => `x = ${num(r.xs[k])} (z = ${num(r.z[k], 2)})`).join(", ") : "ninguno con |z| > 2"}.</li>
      <li><b>Calidad:</b> la recta explica el ${num(100 * q.R2, 2)} % de la variación de y. RMSE = ${num(q.RMSE, 3)} ${esc(UY)}, el
        ${num(100 * q.rmseRel, 2)} % de ȳ y el ${num(100 * q.RMSE / q.rango, 2)} % del rango de y. Un R² alto no descarta la curvatura: hay que leerlo junto con los residuos.</li></ul>`;
    $("tablaResiduos").innerHTML = `<thead><tr><th>x</th><th>y</th><th>ŷ</th><th>e</th><th>z</th></tr></thead><tbody>` +
      r.xs.map((xv, k) => `<tr><td>${num(xv)}</td><td>${num(r.ys[k])}</td><td>${num(r.ys[k] - r.e[k])}</td><td>${num(r.e[k])}</td><td>${num(r.z[k], 2)}</td></tr>`).join("") + "</tbody>";
  }

  // ── Vista 8 · Comparación ──────────────────────────────────────────────────
  const LIBS = [["polyfit", "numpy.polyfit"], ["lstsq", "numpy.linalg.lstsq"], ["linregress", "scipy linregress"]];
  function filasComparacion() {
    const P = D.python, py = !M.modificado;
    const libs = LIBS.map(([k]) => (py && P.bibliotecas[k]) || null);
    const fila = (nombre, man, js, pyv, k) => {
      const vals = [man, js, py ? pyv : null, ...libs.map(l => (l ? l[k] : null))];
      const ok = vals.filter(v => v !== null && Number.isFinite(v));
      return [nombre, ...vals, ok.length > 1 ? Math.max(...ok) - Math.min(...ok) : null];
    };
    return [
      fila("a", st.manual.a, M.aj.a, P.a, 0),
      fila("b", st.manual.b, M.aj.b, P.b, 1),
      fila("S_mín", st.manual.S, M.Smin, P.S, 2),
    ];
  }
  const CAB_COMP = ["Parámetro", "Analítico (a mano)", "Código propio (navegador)", "Código propio (Python)", ...LIBS.map(l => l[1]), "Diferencia máxima"];

  function vComparacion() {
    const filas = filasComparacion();
    $("tablaComparacion").innerHTML = `<thead><tr>${CAB_COMP.map(c => `<th>${c}</th>`).join("")}</tr></thead><tbody>` +
      filas.map(f => `<tr><td>${f[0]}</td>${f.slice(1, -1).map(v => `<td>${v === null ? "—" : num(v, 8)}</td>`).join("")}<td>${exp(f[f.length - 1])}</td></tr>`).join("") + "</tbody>";
    $("notaComparacion").innerHTML = M.modificado
      ? "Los datos actuales no son los de <code>data/input/datos.csv</code>: las columnas de Python y de las bibliotecas no aplican. Descarga el CSV (vista 1), guárdalo como <code>data/input/datos.csv</code> y ejecuta <code>python src/main.py</code> para regenerarlas."
      : "El código propio no usa ninguna función de regresión de biblioteca. Las diferencias con el valor a mano vienen del redondeo; entre código y bibliotecas son errores de punto flotante.";
  }

  // ── Vista 9 · Análisis adicional ───────────────────────────────────────────
  let clicInfluencia = false;
  function vAdicional() {
    const { x, y } = M;
    // A · sensibilidad
    const sen = R.sensibilidad(x, y, st.delta);
    $("deltaTxt").textContent = `${num(st.delta, 1)} ${UY}`;
    dibujar("gSensibilidad", [
      { type: "bar", x, y: sen.map(v => v.da), name: `Δa* [${UA}]`, marker: { color: color("usr") } },
      { type: "scatter", mode: "markers+lines", x, y: sen.map(v => v.db), name: `Δb* [${UY}]`, yaxis: "y2", marker: { color: color("neg") } },
    ], lay(`Cambio de a* y b* al sumar δ = ${num(st.delta, 1)} a cada yᵢ`, EJE_X, "Δa*",
           { yaxis2: { overlaying: "y", side: "right", title: { text: "Δb*" }, showgrid: false }, margin: { l: 60, r: 60, t: 44, b: 50 } }));
    const kmax = sen.reduce((k, v, i) => (Math.abs(v.da) > Math.abs(sen[k].da) ? i : k), 0);
    $("notaSens").textContent = `|Δa*| máximo = ${num(Math.abs(sen[kmax].da))} en x = ${num(x[kmax])}. a* es lineal en y (a* = Σwᵢyᵢ): pesan más los extremos y casi nada el centro.`;

    // B · influencia (clic en una barra = excluir el punto)
    const inf = M.infl, imax = inf.reduce((k, f, i) => (f.dpred > inf[k].dpred ? i : k), 0);
    dibujar("gInfluencia", [{ type: "bar", x: inf.map(f => f.x), y: inf.map(f => f.dpred), name: "máx |Δŷ|",
      marker: { color: inf.map((_, i) => (i === imax ? color("opt") : color("acento"))) },
      customdata: inf.map(f => [f.da, f.db]),
      hovertemplate: `x = %{x}<br>máx |Δŷ| = %{y:.4f} ${UY}<br>Δa = %{customdata[0]:.4f}<br>Δb = %{customdata[1]:.3f}<extra>clic: excluir</extra>` }],
      lay("Cambio máximo de la predicción al eliminar cada punto", EJE_X, `máx |Δŷ| [${UY}]`));
    $("notaInfl").textContent = `El punto más influyente es x = ${num(inf[imax].x)}, y = ${num(inf[imax].y)} (Δa = ${num(inf[imax].da)}, Δb = ${num(inf[imax].db, 3)}). Haz clic en una barra para excluir ese punto y ver cómo cambian todas las vistas.`;
    if (!clicInfluencia) {
      $("gInfluencia").on("plotly_click", e => {
        const p = e.points[0];
        if (!p || !M || M.act.length <= 3) return;
        M.act[p.pointIndex].activo = false;
        construirTabla(); actualizar();
      });
      clicInfluencia = true;
    }

    // C · lineal vs cuadrático
    const q = M.cuad;
    if (q) {
      const xs = lin(Math.min(...x), Math.max(...x), 200);
      dibujar("gCuadratico", [
        { type: "scatter", mode: "markers", x, y, name: "Datos", marker: { color: color("dato"), size: 7 } },
        { type: "scatter", mode: "lines", x: xs, y: xs.map(v => M.aj.a * v + M.aj.b), name: "Lineal", line: { color: color("opt"), width: 2 } },
        { type: "scatter", mode: "lines", x: xs, y: xs.map(v => q.p * v * v + q.q * v + q.r), name: "Cuadrático", line: { color: color("pos"), width: 2, dash: "dash" } },
      ], lay("y = ax + b  vs  y = px² + qx + r", EJE_X, EJE_Y));
      const P = D.python.cuadratico, pv = !M.modificado && P ? P.p_valor : null;
      $("tablaCuad").innerHTML = `<table><thead><tr><th></th><th>Lineal</th><th>Cuadrático</th></tr></thead><tbody>
        <tr><td>S</td><td>${num(q.Sl)}</td><td>${num(q.Sc)}</td></tr>
        <tr><td>R²</td><td>${num(q.R2l, 5)}</td><td>${num(q.R2c, 5)}</td></tr>
        <tr><td>RMSE [${esc(UY)}]</td><td>${num(q.RMSEl)}</td><td>${num(q.RMSEc)}</td></tr></tbody></table>
        <p class="nota">Cuadrático: ŷ = ${num(q.p, 6)}x² ${conSigno(q.q)}x ${conSigno(q.r)}. Prueba F parcial del término x²: F = ${num(q.F, 1)}${pv !== null ? `, p = ${exp(pv)} (scipy)` : ""}.
        Las predicciones difieren a lo sumo ${num(q.maxDif, 3)} ${esc(UY)}: ${q.F > 10 ? "la curvatura es real y estadísticamente significativa, aunque en la práctica la diferencia es pequeña." : "el término x² no mejora de forma clara el ajuste."}</p>`;
    } else {
      Plotly.purge($("gCuadratico"));
      $("tablaCuad").innerHTML = '<p class="nota">El modelo cuadrático necesita n ≥ 4 y al menos 3 valores de x distintos.</p>';
    }

    // D · efecto del número de datos
    const ef = M.efn;
    dibujar("gEfectoN", [
      { type: "scatter", mode: "markers+lines", x: ef.map(f => f.m), y: ef.map(f => f.aMedia), name: "a* medio ± 1 desv.",
        error_y: { type: "data", array: ef.map(f => f.aDesv), visible: true, color: color("usr") }, marker: { color: color("usr") } },
      { type: "scatter", mode: "lines", x: [ef[0].m, ef[ef.length - 1].m], y: [M.aj.a, M.aj.a], name: "a* con todos los datos", line: { color: color("opt"), dash: "dash" } },
    ], lay("a* según el número de datos m", "m (datos usados)", `a* [${UA}]`));
    $("notaEfectoN").textContent = `La desviación de a* baja de ${num(ef[0].aDesv)} con m = ${ef[0].m} a ${num(ef[ef.length - 2] ? ef[ef.length - 2].aDesv : 0)} con m = ${ef[ef.length - 1].m - 1}: con más datos el ajuste se estabiliza.`;
  }

  // ── Pestañas y renderizado ─────────────────────────────────────────────────
  const VISTAS = { datos: vDatos, analitico: vAnalitico, explorador: vExplorador, superficie: v3D, trazas: vTrazas,
                   hessiano: vHessiano, residuos: vResiduos, comparacion: vComparacion, adicional: vAdicional };
  const CON_AB = ["explorador", "superficie", "trazas"];
  const pestañas = [...document.querySelectorAll('[role="tab"]')];

  function render() {
    $("panelAB").hidden = !CON_AB.includes(st.tab) || !M;
    if (st.tab === "datos") return vDatos();
    if (!M) return;
    if (CON_AB.includes(st.tab)) indicador();
    VISTAS[st.tab]();
  }

  function mostrar(tab) {
    st.tab = tab; guardar("vista", tab);
    for (const b of pestañas) {
      const sel = b.dataset.tab === tab;
      b.setAttribute("aria-selected", sel); b.tabIndex = sel ? 0 : -1;
      $(b.dataset.tab).hidden = !sel;
    }
    render();
  }

  function actualizar() { recalcular(); render(); }

  function paridad() {
    const aj = R.ajuste(D.x, D.y), S = R.S(D.x, D.y, aj.a, aj.b), P = D.python;
    const cerca = (u, v) => Math.abs(u - v) <= 1e-9 * (1 + Math.abs(v));
    const ok = cerca(aj.a, P.a) && cerca(aj.b, P.b) && cerca(S, P.S);
    const el = $("paridad");
    el.className = "insignia " + (ok ? "ok" : "mal");
    el.textContent = ok ? "Paridad Python ↔ navegador ✓" : "⚠ Python y el navegador no coinciden";
    el.title = `Python: a=${P.a}, b=${P.b}, S=${P.S} · Navegador: a=${aj.a}, b=${aj.b}, S=${S}`;
  }

  // ── Eventos ────────────────────────────────────────────────────────────────
  pestañas.forEach((b, i) => {
    b.addEventListener("click", () => mostrar(b.dataset.tab));
    b.addEventListener("keydown", e => {
      const d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (!d) return;
      const sig = pestañas[(i + d + pestañas.length) % pestañas.length];
      sig.focus(); mostrar(sig.dataset.tab);
    });
  });

  $("tablaDatos").addEventListener("change", e => {
    const el = e.target, i = Number(el.dataset.i), k = el.dataset.k;
    if (!k) return;
    if (k === "activo") st.filas[i].activo = el.checked;
    else st.filas[i][k] = el.value === "" ? NaN : Number(el.value);
    actualizar();
  });
  $("tablaDatos").addEventListener("click", e => {
    const b = e.target.closest("[data-borrar]");
    if (!b) return;
    st.filas.splice(Number(b.dataset.borrar), 1);
    construirTabla(); actualizar();
  });
  $("agregarFila").addEventListener("click", () => {
    const u = st.filas[st.filas.length - 1];
    st.filas.push({ x: u ? u.x + 1 : 0, y: u ? u.y : 0, activo: true });
    construirTabla(); actualizar();
  });
  $("restablecer").addEventListener("click", () => {
    st.filas = D.x.map((x, i) => ({ x, y: D.y[i], activo: true })); st.a = NaN;
    construirTabla(); actualizar();
  });
  $("archivoCSV").addEventListener("change", e => {
    const f = e.target.files[0];
    if (!f) return;
    f.text().then(t => {
      try { cargarCSV(t); st.a = NaN; construirTabla(); actualizar(); }
      catch (err) { $("error").hidden = false; $("error").textContent = "No se pudo leer el CSV: " + err.message; }
      e.target.value = "";
    });
  });
  $("descargarDatos").addEventListener("click", () =>
    descargar("datos.csv", [["x", "y"], ...st.filas.filter(f => f.activo).map(f => [f.x, f.y])]));
  $("descargarResiduos").addEventListener("click", () => M &&
    descargar("residuos.csv", [["x", "y", "y_pred", "e", "z"], ...M.res.xs.map((xv, k) => [xv, M.res.ys[k], M.res.ys[k] - M.res.e[k], M.res.e[k], M.res.z[k]])]));
  $("descargarComparacion").addEventListener("click", () => M &&
    descargar("comparacion.csv", [CAB_COMP, ...filasComparacion()]));

  for (const [id, k] of [["manA", "a"], ["manB", "b"], ["manS", "S"]]) {
    $(id).addEventListener("input", e => { st.manual[k] = e.target.value === "" ? NaN : Number(e.target.value); textoManual(); });
  }

  $("rangoA").addEventListener("input", e => moverAB(Number(e.target.value), st.b));
  $("rangoB").addEventListener("input", e => moverAB(st.a, Number(e.target.value)));
  $("numA").addEventListener("change", e => Number.isFinite(+e.target.value) && moverAB(+e.target.value, st.b));
  $("numB").addEventListener("change", e => Number.isFinite(+e.target.value) && moverAB(st.a, +e.target.value));
  $("irOptimo").addEventListener("click", () => M && moverAB(M.aj.a, M.aj.b));
  $("irInicio").addEventListener("click", () => { if (M) { rectaPrueba(); moverAB(st.a, st.b); } });

  $("optLog").addEventListener("change", e => { st.log = e.target.checked; render(); });
  $("optCentrado").addEventListener("change", e => { st.centrado = e.target.checked; render(); });
  $("optZoom").addEventListener("change", e => { st.zoom = Number(e.target.value); actualizar(); });
  document.querySelectorAll("[data-cam]").forEach(b =>
    b.addEventListener("click", () => Plotly.relayout($("g3d"), { "scene.camera": CAMARAS[b.dataset.cam] })));
  $("delta").addEventListener("input", e => { st.delta = Number(e.target.value); render(); });

  $("tema").addEventListener("click", () => {
    const raiz = document.documentElement;
    const oscuro = raiz.dataset.theme ? raiz.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    raiz.dataset.theme = oscuro ? "light" : "dark";
    guardar("tema", raiz.dataset.theme);
    render();
  });

  // ── Inicio ─────────────────────────────────────────────────────────────────
  const tema = leer("tema");
  if (tema === "light" || tema === "dark") document.documentElement.dataset.theme = tema;
  $("subtitulo").textContent = `${D.meta.problema} · Fuente: ${D.meta.fuente} · n = ${D.x.length}`;
  construirTabla();
  recalcular();
  paridad();
  const guardada = leer("vista");
  mostrar(VISTAS[guardada] ? guardada : "datos");
})();
