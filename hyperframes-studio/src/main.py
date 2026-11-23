"""
main.py — PUNTO DE ENTRADA ÚNICO DEL PROYECTO
----------------------------------------------
Regresión lineal por mínimos cuadrados con Cálculo Multivariable.

Al ejecutar `python main.py` (o hacer clic / doble clic en él) se hace TODO:

  1. Carga los datos REALES (data/input/datos.csv, fuente en FUENTE.md)
  2. Solución analítica programada (ecuaciones normales) + Hessiano
  3. Verificación numérica del Hessiano (autovalores, diferencias finitas)
  4. Comparación  manual | código propio | biblioteca (§9)
  5. Métodos numéricos: gradiente simple, gradiente centrado, Newton
  6. Residuos, calidad del ajuste y análisis adicional (§11–13)
  5. LIMPIA results/ y genera TODAS las exportaciones:
       results/images/  → 11 gráficas (una por cada sección del enunciado)
       results/html/    → dashboard + superficie 3D interactiva +
                          convergencia + informe imprimible
       results/excel/   → resultados_experimentales.xlsx (4 hojas)
       results/data/    → parametros.json, predicciones.csv, residuos.csv
       results/video/   → explicacion_optimizacion.mp4 (animación del ajuste)
  6. Exporta los datos al proyecto HyperFrames (data/generated/datos.json|js)

Uso:
    python main.py            → flujo completo
    python main.py --sin-gui  → lo mismo, sin abrir ventanas de gráficas
"""

import sys

# Consolas de Windows (cp1252): forzar UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import os
import json
import csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")            # generación headless (sin ventanas)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import plotly.graph_objects as go
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

from datos import obtener_datos, resumen
from analitico import main_analitico, reporte_analitico
from derivadas import verificar_minimo, hessiana, gradiente as calc_gradiente
from metricas import (imprimir_metricas, r_cuadrado, mse, rmse, mae,
                      analizar_pendiente)
from modelo import prediccion, error_cuadratico
from gradiente import (descenso_gradiente, ajustar_eta, imprimir_resultado_gradiente,
                       metodo_newton)
from datos import ETIQUETA_X, ETIQUETA_Y, PROBLEMA, FUENTE
import analisis
import graficas
import graficas_3d
import modelos_3d
import presentacion

# ── Rutas ────────────────────────────────────────────────────────────────────
RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS = os.path.join(RAIZ, "results")
CARPETA_HYPERFRAMES = os.path.join(RAIZ, "data", "generated")

# ── Paleta (azul académico) ─────────────────────────────────────────────────
AZUL = "#1f5fa8"
AZUL_CLARO = "#7fb2e5"
AZUL_PASTEL = "#eef5fc"
GRIS = "#5a6b7d"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRIS, "axes.labelcolor": "#1a2733",
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "font.family": "DejaVu Sans", "grid.color": "#dbe4ee", "grid.alpha": 0.8,
})

PLANTILLA_PLOTLY = "plotly_white"


# ═══════════ LIMPIEZA DE results/ ═══════════
def _limpiar_results():
    """BORRA el contenido de results/ (las carpetas se mantienen)."""
    eliminados = 0
    for sub in ("pdf", "excel", "html", "images", "data", "video"):
        carpeta = os.path.join(RESULTS, sub)
        os.makedirs(carpeta, exist_ok=True)
        for nombre in os.listdir(carpeta):
            ruta = os.path.join(carpeta, nombre)
            if os.path.isfile(ruta):
                os.remove(ruta)
            else:
                import shutil
                shutil.rmtree(ruta)
            eliminados += 1
    if eliminados:
        print(f"  [x] results/ reiniciado: {eliminados} archivo(s) anterior(es) eliminados")


# ═══════════ data/ ═══════════════════
def exportar_data(x, y, r, h, g, res_metricas):
    carpeta = os.path.join(RESULTS, "data")
    parametros = {
        "modelo": "y = a·x + b (mínimos cuadrados)",
        "n": r["n"],
        "pendiente_optima": r["a_star"],
        "intercepto_optimo": r["b_star"],
        "ecuacion": f"y = {r['a_star']:.4f}x + {r['b_star']:.4f}",
        "error_cuadratico_minimo": r["s_min"],
        "sumatorias": {"sx": r["sum_x"], "sy": r["sum_y"],
                       "sxy": r["sum_xy"], "sxx": r["sum_x2"]},
        "medias": {"x": r["x_mean"], "y": r["y_mean"]},
        "hessiana": h["H"].tolist(),
        "det_hessiana": h["det_H"],
        "es_minimo_global": bool(h["es_minimo"]),
        "gradiente": {
            "a": g["a_star"], "b": g["b_star"],
            "iteraciones": g["iteraciones"], "convergio": bool(g["convergio"]),
            "eta": ajustar_eta(x),
        },
    }
    with open(os.path.join(carpeta, "parametros.json"), "w", encoding="utf-8") as f:
        json.dump(parametros, f, indent=2, ensure_ascii=False)

    y_hat = prediccion(x, r["a_star"], r["b_star"])

    with open(os.path.join(carpeta, "predicciones.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "y_real", "y_predicho", "metodo"])
        for xi, yi, yp in zip(x, y, y_hat):
            w.writerow([f"{xi:.6f}", f"{yi:.6f}", f"{yp:.6f}", "minimos_cuadrados"])

    with open(os.path.join(carpeta, "residuos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "residuo_e", "residuo_cuadrado", "signo"])
        for xi, yi, yp in zip(x, y, y_hat):
            e = yi - yp
            w.writerow([f"{xi:.6f}", f"{e:.6f}", f"{e*e:.6f}",
                        "positivo" if e > 0 else ("negativo" if e < 0 else "cero")])


# ═══════════ excel/ ═══════════════════
def exportar_excel(x, y, r, h, g, comp=None):
    ruta = os.path.join(RESULTS, "excel", "resultados_experimentales.xlsx")
    wb = openpyxl.Workbook()
    enc = Font(bold=True, color="FFFFFF")
    relleno = PatternFill("solid", fgColor="1F5FA8")
    titulo = Font(bold=True, size=13, color="1F5FA8")

    def encabezado(ws, fila, cols):
        for j, c in enumerate(cols, start=1):
            cell = ws.cell(row=fila, column=j, value=c)
            cell.font = enc
            cell.fill = relleno
            cell.alignment = Alignment(horizontal="center")

    ws = wb.active
    ws.title = "Datos"
    ws["A1"] = f"Datos reales: {PROBLEMA}"
    ws["A1"].font = titulo
    encabezado(ws, 3, ["i", ETIQUETA_X, ETIQUETA_Y])
    for i, (xi, yi) in enumerate(zip(x, y), start=1):
        ws.cell(row=3 + i, column=1, value=i)
        ws.cell(row=3 + i, column=2, value=float(xi))
        ws.cell(row=3 + i, column=3, value=float(yi))
    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 14

    ws = wb.create_sheet("Ajuste")
    ws["A1"] = "Parámetros óptimos"
    ws["A1"].font = titulo
    filas = [
        ("Pendiente a*", r["a_star"]),
        ("Intercepto b*", r["b_star"]),
        ("S(a*, b*) mínimo", r["s_min"]),
        ("Iteraciones gradiente", g["iteraciones"]),
        ("Gradiente a*", g["a_star"]),
        ("Gradiente b*", g["b_star"]),
        ("Convergió", g["convergio"]),
    ]
    encabezado(ws, 3, ["Parámetro", "Valor"])
    for i, (k, v) in enumerate(filas, start=4):
        ws.cell(row=i, column=1, value=k)
        ws.cell(row=i, column=2, value=v)
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 18

    ws = wb.create_sheet("Residuos")
    y_hat = prediccion(x, r["a_star"], r["b_star"])
    ws["A1"] = "Análisis de residuos"
    ws["A1"].font = titulo
    encabezado(ws, 3, ["x", "y_real", "y_predicho", "e_i", "e_i²"])
    for i, (xi, yi, yp) in enumerate(zip(x, y, y_hat), start=4):
        e = yi - yp
        ws.cell(row=i, column=1, value=float(xi))
        ws.cell(row=i, column=2, value=float(yi))
        ws.cell(row=i, column=3, value=float(yp))
        ws.cell(row=i, column=4, value=float(e))
        ws.cell(row=i, column=5, value=float(e * e))
    for col in "ABCDE":
        ws.column_dimensions[col].width = 13

    ws = wb.create_sheet("Hessiana")
    ws["A1"] = "Matriz Hessiana y verificación de mínimo"
    ws["A1"].font = titulo
    H = h["H"]
    ws.cell(row=3, column=1, value="H").font = titulo
    for i in range(2):
        for j in range(2):
            ws.cell(row=4 + i, column=2 + j, value=float(H[i, j]))
    ws.cell(row=7, column=1, value="det(H)")
    ws.cell(row=7, column=2, value=h["det_H"])
    ws.cell(row=8, column=1, value="¿Mínimo global?")
    ws.cell(row=8, column=2, value="SÍ" if h["es_minimo"] else "NO")
    ws.column_dimensions["A"].width = 20

    if comp is not None:
        ws = wb.create_sheet("Comparación")
        ws["A1"] = "Comparación analítico (manual) / código propio / biblioteca"
        ws["A1"].font = titulo
        encabezado(ws, 3, ["Parámetro", "Analítico", "Código propio", "Biblioteca", "Δ máx"])
        for i, (nom, m, cod, lib, d3, _d2) in enumerate(comp["filas"], start=4):
            for j, v in enumerate([nom, m, cod, lib, d3], start=1):
                ws.cell(row=i, column=j, value=v)
        for col in "ABCDE":
            ws.column_dimensions[col].width = 18

    wb.save(ruta)
    return ruta


# ═══════════ html/ (Plotly) ═══════════════════
def exportar_html(x, y, r, h, g):
    carpeta = os.path.join(RESULTS, "html")
    a, b = r["a_star"], r["b_star"]

    # superficie_error.html — superficie 3D interactiva
    (a0, a1), (b0, b1) = graficas.ventana(x, a, b, 6 * r["s_min"])
    A = np.linspace(a0, a1, 70)
    B = np.linspace(b0, b1, 70)
    AA, BB = np.meshgrid(A, B)
    S = np.array([[error_cuadratico(x, y, av, bv) for av in A] for bv in B])

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=A, y=B, z=S, colorscale="Blues", opacity=0.92,
        colorbar=dict(title="S(a,b)"),
    ))
    fig.add_trace(go.Scatter3d(
        x=[a], y=[b], z=[r["s_min"]],
        mode="markers+text", marker=dict(color="#d64545", size=9),
        text=["Mínimo global"], textposition="top center", name="Mínimo",
    ))
    fig.update_layout(
        template=PLANTILLA_PLOTLY,
        title="Superficie del error S(a,b) — arrastra para girar · scroll para zoom",
        scene=dict(
            xaxis_title="a (pendiente)", yaxis_title="b (intercepto)", zaxis_title="S(a,b)",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        autosize=True,
    )
    fig.write_html(os.path.join(carpeta, "superficie_error.html"), include_plotlyjs="cdn")

    # convergencia.html
    hist = g["historial"]
    iters = [hh[0] for hh in hist]
    Svals = [hh[1] for hh in hist]
    normas = [hh[2] for hh in hist]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iters, y=Svals, mode="lines+markers",
                             name="S(a,b)", line=dict(color=AZUL)))
    fig.add_trace(go.Scatter(x=iters, y=normas, mode="lines+markers",
                             name="||grad S||", line=dict(color="#d64545"), yaxis="y2"))
    fig.update_layout(
        template=PLANTILLA_PLOTLY,
        title="Convergencia del descenso por gradiente",
        xaxis_title="Iteración", yaxis=dict(title="S(a,b)"),
        yaxis2=dict(title="||grad S||", overlaying="y", side="right", type="log"),
        hovermode="x unified",
    )
    fig.write_html(os.path.join(carpeta, "convergencia.html"), include_plotlyjs="cdn")

    # dashboard.html
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Datos",
                             marker=dict(color=AZUL, size=10)))
    xs = np.linspace(x.min(), x.max(), 50)
    fig.add_trace(go.Scatter(x=xs, y=prediccion(xs, a, b), mode="lines",
                             name=f"y = {a:.3f}x + {b:.3f}", line=dict(color="#d64545")))
    fig.update_layout(template=PLANTILLA_PLOTLY,
                      title="Ajuste por mínimos cuadrados",
                      xaxis_title=ETIQUETA_X, yaxis_title=ETIQUETA_Y)
    fig.write_html(os.path.join(carpeta, "dashboard.html"),
                   include_plotlyjs="cdn", full_html=True)

    # informe imprimible (Ctrl+P → informe_completo.pdf)
    with open(os.path.join(carpeta, "informe_imprimible.html"), "w",
              encoding="utf-8") as f:
        f.write(crear_informe_html(x, y, r, h, g))


def crear_informe_html(x, y, r, h, g):
    a, b = r["a_star"], r["b_star"]
    y_hat = prediccion(x, a, b)
    filas = "".join(
        f"<tr><td>{i+1}</td><td>{xi:.3f}</td><td>{yi:.3f}</td>"
        f"<td>{yh:.3f}</td><td>{yi-yh:.3f}</td></tr>"
        for i, (xi, yi, yh) in enumerate(zip(x, y, y_hat))
    )
    informe = f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<title>Informe completo — Mínimos cuadrados</title>
<style>
body{{font-family:Georgia,serif;max-width:800px;margin:40px auto;color:#1a2733}}
h1{{color:{AZUL}}} h2{{color:{AZUL};border-bottom:2px solid {AZUL_CLARO};padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;margin:14px 0}}
td,th{{border:1px solid #c9d6e5;padding:8px;text-align:left}}
th{{background:{AZUL};color:white}}
tr:nth-child(even) td{{background:{AZUL_PASTEL}}}
.ecuacion{{text-align:center;font-size:1.3em;background:{AZUL_PASTEL};padding:12px;border-radius:8px}}
@media print{{body{{margin:15mm}}}}
</style></head><body>
<h1>Informe completo — Regresión lineal por mínimos cuadrados</h1>
<h2>1. Datos y estadísticas</h2>
<p>n = {r['n']} observaciones · x̄ = {r['x_mean']:.4f} · ȳ = {r['y_mean']:.4f}</p>
<table><tr><th>i</th><th>xᵢ</th><th>yᵢ</th><th>ŷᵢ</th><th>eᵢ</th></tr>{filas}</table>
<h2>2. Parámetros óptimos</h2>
<table>
<tr><th>Parámetro</th><th>Valor</th></tr>
<tr><td>Pendiente a*</td><td>{a:.6f}</td></tr>
<tr><td>Intercepto b*</td><td>{b:.6f}</td></tr>
<tr><td>S(a*, b*) mínimo</td><td>{r['s_min']:.6f}</td></tr>
<tr><td>det(H)</td><td>{h['det_H']:.4f} (&gt; 0 → mínimo global)</td></tr>
<tr><td>Gradiente: a*, b* ({g['iteraciones']} iter.)</td><td>{g['a_star']:.6f}, {g['b_star']:.6f}</td></tr>
</table>
<h2>3. Recta de regresión</h2>
<p class="ecuacion">ŷ = {a:.4f}·x + ({b:.4f})</p>
<h2>4. Conclusiones</h2>
<ul>
<li>Problema: {PROBLEMA}. Fuente: {FUENTE}.</li>
<li>La Hessiana es definida positiva: (a*, b*) es el <b>mínimo global</b> de S.</li>
<li>Cada pulgada adicional de estatura se asocia con {a:.2f} lb más de peso promedio.</li>
<li>Los residuos forman una U (positivos en los extremos, negativos en el centro):
la relación es ligeramente curva y un modelo cuadrático la describe mejor.</li>
<li>Ver docs/Conclusiones.md para el análisis completo.</li>
</ul>
<p><i>Generado automáticamente por main.py · Imprimir con Ctrl+P para obtener informe_completo.pdf</i></p>
</body></html>"""
    return informe


# ═══════════ video/ ═══════════
def crear_video(x, y, r, g, trayectoria, fps=12):
    """MP4 con la recta ajustándose por gradiente + trayectoria en el contorno."""
    try:
        import imageio
    except ImportError:
        print("  [!] video omitido: instala con  python -m pip install imageio imageio-ffmpeg")
        return None

    carpeta = os.path.join(RESULTS, "video")
    ruta = os.path.join(carpeta, "explicacion_optimizacion.mp4")
    a, b = r["a_star"], r["b_star"]

    tray = trayectoria if trayectoria and len(trayectoria) > 1 else [(0.0, 0.0, 0, 0, 0), (a, b, 0, 0, 1)]
    idx = np.linspace(0, len(tray) - 1, min(60, len(tray))).astype(int)
    tray_s = [tray[i] for i in idx]

    A = np.linspace(-0.6, 4.3, 100)
    B = np.linspace(-105, 15, 100)
    AA, BB = np.meshgrid(A, B)
    S = np.array([[error_cuadratico(x, y, av_, bv_) for av_ in A] for bv_ in B])

    frames = []
    xs = np.linspace(x.min(), x.max(), 50)
    for k, (ak, bk, *_resto) in enumerate(tray_s):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.scatter(x, y, color=AZUL, s=50, zorder=3, label="Datos")
        ax1.set_ylim(y.min() - 15, y.max() + 15)
        ax1.plot(xs, prediccion(xs, ak, bk), color="#d64545", lw=2,
                 label=f"iter {tray_s[k][4]}: y={ak:.3f}x{bk:+.2f}")
        ax1.set_title(f"Ajuste por gradiente (x centrada) — iteración {tray_s[k][4]}")
        ax1.set_xlabel(ETIQUETA_X); ax1.set_ylabel(ETIQUETA_Y); ax1.legend(loc="upper left")
        ax1.grid(True)

        ax2.contourf(AA, BB, np.log10(S), levels=25, cmap="Blues_r")
        ax2.plot([p[0] for p in tray_s[:k+1]], [p[1] for p in tray_s[:k+1]],
                 "-o", color="#d64545", markersize=3, lw=1.5, label="trayectoria")
        ax2.plot(a, b, "*", color=AZUL, markersize=18, label="mínimo global")
        ax2.set_title("S(a, b): curvas de nivel")
        ax2.set_xlabel("a (pendiente)"); ax2.set_ylabel("b (intercepto)")
        ax2.legend(loc="upper right")

        fig.tight_layout()
        fig.canvas.draw()
        import PIL.Image as Image
        img = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                               fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
        frames.append(np.asarray(img.convert("RGB")))
        plt.close(fig)

    frames.extend([frames[-1]] * fps)
    try:
        imageio.mimsave(ruta, frames, fps=fps, codec="libx264",
                        macro_block_size=1, quality=8)
    except TypeError:
        imageio.mimsave(ruta, frames, fps=fps)
    print(f"  [>] video generado: {ruta}  ({len(frames)} frames @ {fps} fps)")
    return ruta


# ═══════════ Exportación a HyperFrames ═══════════════════
def exportar_hyperframes(x, y, r, h, g, res_metricas) -> dict:
    """Escribe data/generated/datos.json y datos.js para el video."""
    a, b = r["a_star"], r["b_star"]
    y_hat = prediccion(x, a, b)

    resultado = {
        "total": int(r["n"]),
        "pendiente": a,
        "intercepto": b,
        "errorCuadratico": r["s_min"],
        "ecuacion": f"ŷ = {a:.4f}·x {b:+.4f}",
        "problema": PROBLEMA,
        "fuente": FUENTE,
        "etiquetaX": ETIQUETA_X,
        "etiquetaY": ETIQUETA_Y,
        "sumas": {"x": r["sum_x"], "y": r["sum_y"], "xy": r["sum_xy"], "xx": r["sum_x2"]},
        "medias": {"x": r["x_mean"], "y": r["y_mean"]},
        "r2": r_cuadrado(y, y_hat),
        "mse": mse(y, y_hat),
        "rmse": rmse(y, y_hat),
        "mae": mae(y, y_hat),
        "relacion": analizar_pendiente(a),
        "hessiana": [[float(h["H"][0, 0]), float(h["H"][0, 1])],
                     [float(h["H"][1, 0]), float(h["H"][1, 1])]],
        "detHessiana": float(h["det_H"]),
        "esMinimoGlobal": bool(h["es_minimo"]),
        "gradiente": {
            "a": g["a_star"],
            "b": g["b_star"],
            "iteraciones": g["iteraciones"],
            "convergio": bool(g["convergio"]),
            "eta": g.get("eta"),
            "centrado": bool(g.get("centrado", False)),
        },
        "datos": [{"x": float(xi), "y": float(yi)} for xi, yi in zip(x, y)],
    }

    os.makedirs(CARPETA_HYPERFRAMES, exist_ok=True)
    with open(os.path.join(CARPETA_HYPERFRAMES, "datos.json"), "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    with open(os.path.join(CARPETA_HYPERFRAMES, "datos.js"), "w", encoding="utf-8") as f:
        f.write(f"window.HYPEFRAMES_DATA = {json.dumps(resultado, indent=2, ensure_ascii=False)};\n")

    print("\n" + "═" * 55)
    print("  EXPORTACIÓN A HYPERFRAMES")
    print("═" * 55)
    print(f"  total           = {resultado['total']}")
    print(f"  pendiente a*    = {resultado['pendiente']:.6f}")
    print(f"  intercepto b*   = {resultado['intercepto']:.6f}")
    print(f"  S(a*, b*)       = {resultado['errorCuadratico']:.6f}")
    print(f"  R²              = {resultado['r2']:.6f}")
    print(f"   -> {os.path.join(CARPETA_HYPERFRAMES, 'datos.json')}")
    print(f"   -> {os.path.join(CARPETA_HYPERFRAMES, 'datos.js')}")
    print("═" * 55)
    return resultado


# ═══════════════════ FLUJO PRINCIPAL ═══════════════════
def main(mostrar_graficas: bool = True) -> None:
    print("╔" + "═" * 58 + "╗")
    print("║  REGRESIÓN LINEAL POR MÍNIMOS CUADRADOS                        ║")
    print("║  Cálculo Multivariable — derivadas parciales y Hessiana        ║")
    print("╚" + "═" * 58 + "╝")

    # ── 1. Datos ────────────────────────────────────────────────────────
    x, y = obtener_datos()
    resumen(x, y)

    # ── 2-4. Solución analítica + Hessiana ──────────────────────────────
    x, y, r = main_analitico()
    a_star, b_star = r["a_star"], r["b_star"]
    h = verificar_minimo(x)

    # ── 3. Hessiano: verificación numérica (§8.6–8.7) ───────────────────
    hv = analisis.verificar_hessiano_numerico(x, y, a_star, b_star)
    analisis.imprimir_hessiano_numerico(hv)

    # ── 4. Comparación manual / código propio / biblioteca (§9) ─────────
    comp = analisis.comparar_metodos(x, y)
    analisis.imprimir_comparacion(comp)

    # ── 5. Métodos numéricos (complemento) ──────────────────────────────
    gd_simple = descenso_gradiente(x, y, max_iter=200_000)
    imprimir_resultado_gradiente(gd_simple, "DESCENSO POR GRADIENTE SIMPLE en (a, b)")
    gd_centrado = descenso_gradiente(x, y, centrar=True, max_iter=200_000)
    imprimir_resultado_gradiente(gd_centrado, "DESCENSO POR GRADIENTE con x centrada")
    newton = metodo_newton(x, y)
    print(f"\n• Método de Newton (1 paso desde (0,0)): a = {newton['a_star']:.10f}, "
          f"b = {newton['b_star']:.10f}, ‖∇S‖ = {newton['norma_final']:.1e}")
    print(f"  El gradiente simple se detuvo en ({gd_simple['a_star']:.3f}, {gd_simple['b_star']:.2f}) "
          f"tras {gd_simple['iteraciones']} iteraciones: κ(H) = {hv['condicion']:.1e}.")
    res_grad = gd_centrado

    # ── 6. Residuos, calidad y análisis adicional (§11–13) ──────────────
    res_metricas = imprimir_metricas(x, y, a_star, b_star)
    resid = analisis.analizar_residuos(x, y, a_star, b_star)
    calidad = analisis.calidad_ajuste(x, y, a_star, b_star)
    analisis.imprimir_residuos_calidad(resid, calidad)
    inf = analisis.influencia_datos(x, y)
    cq = analisis.comparar_cuadratico(x, y)
    sen = analisis.sensibilidad(x, y)
    analisis.imprimir_adicional(inf, cq, sen)

    # ── 7. REINICIAR results/ y generar TODAS las exportaciones ─────────
    print("\n" + "═" * 55)
    print("  GENERACIÓN DE RESULTADOS  (results/)")
    print("═" * 55)
    _limpiar_results()
    for sub in ("pdf", "excel", "html", "images", "data", "video"):
        os.makedirs(os.path.join(RESULTS, sub), exist_ok=True)

    # 7a. Gráficas avanzadas → results/images/
    print("\n• Gráficas (11 PNG, una por sección del enunciado):")
    graficas.generar_todas(x, y, a_star, b_star, resid, gd_simple, gd_centrado,
                           newton, inf, cq)
    graficas_3d.generar_graficas_3d(x, y, a_star, b_star, gd_simple, gd_centrado, newton)
    print(f"   -> {os.path.join(RESULTS, 'images')}")

    # 7b. Dashboards interactivos + informe → results/html/
    print("\n• Dashboards interactivos (Plotly) e informe imprimible:")
    exportar_html(x, y, r, h, res_grad)
    ruta_3d = modelos_3d.generar_modelos_3d(x, y, a_star, b_star, gd_simple, gd_centrado,
                                            newton, hv, calidad)
    print(f"   -> modelos 3D interactivos: {ruta_3d}")
    print(f"   -> presentación interactiva: {presentacion.construir()}")
    print(f"   -> {os.path.join(RESULTS, 'html')}")

    # 7c. Excel → results/excel/
    print("\n• Excel (openpyxl, 4 hojas):")
    ruta_excel = exportar_excel(x, y, r, h, res_grad, comp)
    print(f"   -> {ruta_excel}")

    # 7d. Datos crudos → results/data/
    print("\n• Datos crudos (JSON + CSV):")
    exportar_data(x, y, r, h, res_grad, res_metricas)
    print(f"   -> {os.path.join(RESULTS, 'data')}")

    # 7e. Video MP4 → results/video/
    print("\n• Video de explicación (MP4):")
    crear_video(x, y, r, res_grad, res_grad["trayectoria"])

    # ── 8. Exportar a HyperFrames ───────────────────────────────────────
    exportar_hyperframes(x, y, r, h, res_grad, res_metricas)
    print(f"\n[OK] results/ regenerado completo en: {os.path.abspath(RESULTS)}")

    # ── Conclusión ──────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  CONCLUSIÓN")
    print("═" * 60)
    print(f"""
  Problema: {PROBLEMA}
  La recta óptima  ŷ = {a_star:.4f}·x + ({b_star:.4f})  minimiza el error
  cuadrático total  S(a*, b*) = {r['s_min']:.4f} lb²  (R² = {calidad['R2']:.4f},
  RMSE = {calidad['RMSE']:.3f} lb).

  • Hessiano definido positivo (λ₁, λ₂ > 0) ⇒ (a*, b*) es el MÍNIMO GLOBAL.
  • Manual, código propio y biblioteca coinciden (diferencias de redondeo).
  • Los residuos forman una U ⇒ la relación real es levemente curva; el modelo
    cuadrático reduce S de {cq['S_lineal']:.2f} a {cq['S_cuad']:.2f}, aunque sus
    predicciones difieren a lo sumo {cq['max_dif_pred']:.2f} lb.
""")
    print("═" * 60)


if __name__ == "__main__":
    sin_gui = "--sin-gui" in sys.argv
    main(mostrar_graficas=not sin_gui)
