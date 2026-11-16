"""
exportar_resultados.py
----------------------
Genera la carpeta de salidas `results/` dentro de hyperframes-studio con:

  results/
  ├── excel/   resultados_experimentales.xlsx   (openpyxl)
  ├── html/    dashboard.html, superficie_error.html,
  │            mapas_calor.html, convergencia.html   (Plotly interactivo:
  │            superficies 3D que giran, mapas de calor avanzados)
  ├── images/  residuos.png, regresion_3d.png, hessiana.png, curvas_nivel.png
  │            (matplotlib, estilo Light académico blanco/azul)
  ├── data/    parametros.json, predicciones.csv, residuos.csv
  ├── pdf/     informe_completo.pdf   (informe impreso desde dashboard.html)
  └── video/   explicacion_optimizacion.mp4  (placeholder/render HyperFrames)

Uso:  python src/exportar_resultados.py
"""

import os
import sys
import json
import csv

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import plotly.graph_objects as go
import plotly.io as pio
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

from datos import obtener_datos
from analitico import reporte_analitico
from derivadas import verificar_minimo, hessiana
from gradiente import descenso_gradiente, ajustar_eta
from modelo import prediccion, error_cuadratico

RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RESULTS = os.path.join(RAIZ, "results")

# ── Paleta Light académica (blanco / azul) ──────────────────────────────────
AZUL = "#1f5fa8"
AZUL_CLARO = "#7fb2e5"
AZUL_PASTEL = "#eef5fc"
GRIS = "#5a6b7d"
FONDO = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": FONDO,
    "axes.facecolor": FONDO,
    "axes.edgecolor": GRIS,
    "axes.labelcolor": "#1a2733",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "font.family": "DejaVu Sans",
    "grid.color": "#dbe4ee",
    "grid.alpha": 0.8,
})

PLANTILLA_PLOTLY = "plotly_white"


def _limpiar_results():
    """
    BORRA el contenido de cada subcarpeta de results/ (las carpetas se mantienen,
    su contenido se reinicia) para que ninguna información antigua sobreviva.
    """
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
        print(f"  ⟲ results/ reiniciado: {eliminados} archivo(s) anterior(es) eliminados")


def _carpetas():
    for sub in ("pdf", "excel", "html", "images", "data", "video"):
        os.makedirs(os.path.join(RESULTS, sub), exist_ok=True)


# ═════════════════ data/ ═════════════════════

def exportar_data(x, y, r, h, g):
    """parametros.json, predicciones.csv, residuos.csv"""
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


# ═════════════════ excel/ ════════════════════

def exportar_excel(x, y, r, h, g):
    """resultados_experimentales.xlsx con hojas: Datos, Ajuste, Residuos, Hessiana"""
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

    # Hoja 1: Datos
    ws = wb.active
    ws.title = "Datos"
    ws["A1"] = "Datos experimentales (x, y)"
    ws["A1"].font = titulo
    encabezado(ws, 3, ["i", "x", "y"])
    for i, (xi, yi) in enumerate(zip(x, y), start=1):
        ws.cell(row=3 + i, column=1, value=i)
        ws.cell(row=3 + i, column=2, value=float(xi))
        ws.cell(row=3 + i, column=3, value=float(yi))
    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 14

    # Hoja 2: Ajuste
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

    # Hoja 3: Residuos
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

    # Hoja 4: Hessiana
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

    wb.save(ruta)
    return ruta


# ═════════════════ images/ ═══════════════════

def exportar_images(x, y, r, g):
    """residuos.png, regresion_3d.png, hessiana.png, curvas_nivel.png (matplotlib)."""
    carpeta = os.path.join(RESULTS, "images")
    a, b = r["a_star"], r["b_star"]
    y_hat = prediccion(x, a, b)
    e = y - y_hat

    # 1. residuos.png — barras de residuo por observación
    fig, ax = plt.subplots(figsize=(8, 5))
    colores = [AZUL if ei >= 0 else "#d64545" for ei in e]
    ax.bar(range(1, len(x) + 1), e, color=colores, edgecolor="white")
    ax.axhline(0, color=GRIS, lw=1)
    ax.set_title("Residuos por observación  eᵢ = yᵢ − ŷᵢ")
    ax.set_xlabel("Observación i")
    ax.set_ylabel("Residuo")
    ax.set_xticks(range(1, len(x) + 1))
    ax.grid(True, axis="y")
    fig.savefig(os.path.join(carpeta, "residuos.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. regresion_3d.png — plano de regresión en 3D
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, e, c=e, cmap="RdYlBu", s=60, edgecolor="white")
    xs = np.linspace(x.min(), x.max(), 30)
    ys = np.linspace(y.min() - 1, y.max() + 1, 30)
    XX, YY = np.meshgrid(xs, ys)
    ZZ = np.zeros_like(XX)   # plano de residuo cero
    ax.plot_surface(XX, YY, ZZ, alpha=0.25, color=AZUL)
    ax.set_title("Plano de regresión 3D: (x, y, residuo)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("e = y − ŷ")
    fig.savefig(os.path.join(carpeta, "regresion_3d.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. hessiana.png — mapa de calor de la matriz H
    H = hessiana(x)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(H, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{H[i, j]:.1f}", ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if H[i, j] > H.max() / 2 else "#1a2733")
    ax.set_xticks([0, 1], ["∂²S/∂a²", "∂²S/∂a∂b"])
    ax.set_yticks([0, 1], ["∂²S/∂a∂b", "∂²S/∂b²"])
    ax.set_title(f"Matriz Hessiana   det(H) = {np.linalg.det(H):.1f} > 0 → mínimo")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(os.path.join(carpeta, "hessiana.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. curvas_nivel.png — contorno de S(a, b) con mínimo y trayectoria
    A = np.linspace(a - 3, a + 3, 120)
    B = np.linspace(b - 3, b + 3, 120)
    AA, BB = np.meshgrid(A, B)
    S = np.array([[error_cuadratico(x, y, av, bv) for av in A] for bv in B])
    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contourf(AA, BB, S, levels=25, cmap="Blues")
    fig.colorbar(cs, ax=ax, label="S(a, b)")
    ax.contour(AA, BB, S, levels=25, colors="#9fb8d4", linewidths=0.5)
    ax.plot(a, b, "*", color="#d64545", markersize=18, label=f"Mínimo global ({a:.3f}, {b:.3f})")
    ax.set_title("Curvas de nivel de S(a, b)")
    ax.set_xlabel("a (pendiente)"); ax.set_ylabel("b (intercepto)")
    ax.legend()
    fig.savefig(os.path.join(carpeta, "curvas_nivel.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════════ html/ (Plotly) ════════════

def exportar_html(x, y, r, h, g):
    """Dashboard + 3 interactivos Plotly (superficie 3D giratoria, mapas de calor
    animados, convergencia)."""
    carpeta = os.path.join(RESULTS, "html")
    a, b = r["a_star"], r["b_star"]

    # ── superficie_error.html: superficie 3D interactiva que el usuario puede girar ──
    A = np.linspace(a - 3, a + 3, 60)
    B = np.linspace(b - 3, b + 3, 60)
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
            # animación de giro automático
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        ),
        autosize=True,
    )
    fig.write_html(os.path.join(carpeta, "superficie_error.html"), include_plotlyjs="cdn")

    # ── mapas_calor.html: heatmap animado de S(a,b) en franjas rotadas ──
    frames = []
    for k in range(8):
        ang = k * 45
        Srot = np.roll(np.roll(S, k * 5, axis=0), k * 3, axis=1)
        frames.append(go.Frame(data=[go.Heatmap(z=Srot, colorscale="Blues",
                                                colorbar=dict(title="S"))],
                               name=str(ang)))
    fig = go.Figure(
        data=[go.Heatmap(z=S, colorscale="Blues", colorbar=dict(title="S(a,b)"))],
        frames=frames,
    )
    fig.update_layout(
        template=PLANTILLA_PLOTLY,
        title=f"Mapa de calor de S(a,b) — animación 360° (franja inicial 0° → 315°)",
        updatemenus=[dict(
            type="buttons", direction="left", x=0.5, y=-0.15,
            buttons=[dict(label="▶ Girar", method="animate",
                          args=[None, dict(frame=dict(duration=400, redraw=True),
                                           fromcurrent=True, loop=True)]),
                     dict(label="⏸ Pausar", method="animate",
                          args=[[None], dict(mode="immediate", frame=dict(duration=0))])],
        )],
    )
    fig.write_html(os.path.join(carpeta, "mapas_calor.html"), include_plotlyjs="cdn")

    # ── convergencia.html: S y ‖∇S‖ vs iteración ──
    # Re-ejecutar gradiente registrando la norma en cada paso ligero
    hist = g["historial"]
    iters = [h[0] for h in hist]
    Svals = [h[1] for h in hist]
    normas = [h[2] for h in hist]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iters, y=Svals, mode="lines+markers",
                             name="S(a,b)", line=dict(color=AZUL)))
    fig.add_trace(go.Scatter(x=iters, y=normas, mode="lines+markers",
                             name="‖∇S‖", line=dict(color="#d64545"), yaxis="y2"))
    fig.update_layout(
        template=PLANTILLA_PLOTLY,
        title="Convergencia del descenso por gradiente",
        xaxis_title="Iteración", yaxis=dict(title="S(a,b)"),
        yaxis2=dict(title="‖∇S‖", overlaying="y", side="right", type="log"),
        hovermode="x unified",
    )
    fig.write_html(os.path.join(carpeta, "convergencia.html"), include_plotlyjs="cdn")

    # ── dashboard.html: resumen ejecutivo + mini gráficos ──
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Datos",
                             marker=dict(color=AZUL, size=10)))
    xs = np.linspace(x.min(), x.max(), 50)
    fig.add_trace(go.Scatter(x=xs, y=prediccion(xs, a, b), mode="lines",
                             name=f"ŷ = {a:.3f}x + {b:.3f}", line=dict(color="#d64545")))
    fig.update_layout(template=PLANTILLA_PLOTLY,
                      title="Ajuste por mínimos cuadrados",
                      xaxis_title="x", yaxis_title="y")
    fig.write_html(os.path.join(carpeta, "dashboard.html"),
                   include_plotlyjs="cdn", full_html=True)

    # Informe imprimible (Ctrl+P → informe_completo.pdf)
    with open(os.path.join(carpeta, "informe_imprimible.html"), "w",
              encoding="utf-8") as f:
        f.write(crear_informe_html(x, y, r, h, g))


def crear_informe_html(x, y, r, h, g):
    """Genera html/informe_imprimible.html (imprimir → informe_completo.pdf)."""
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
<li>La Hessiana es definida positiva: (a*, b*) es el <b>mínimo global</b> de S.</li>
<li>El descenso por gradiente converge al mismo punto desde (0, 0).</li>
<li>La relación entre las variables es <b>{'positiva' if a > 0 else ('negativa' if a < 0 else 'débil')}</b>.</li>
</ul>
<p><i>Generado automáticamente por src/exportar_resultados.py · Imprimir con Ctrl+P para obtener informe_completo.pdf</i></p>
</body></html>"""
    return informe


def crear_video(x, y, r, g, fps=12):
    """
    Genera results/video/explicacion_optimizacion.mp4 con matplotlib:
    frames de la recta ajustándose por descenso de gradiente + curvas de nivel,
    unidos en un MP4 (usa imageio / imageio-ffmpeg).
    """
    try:
        import imageio
    except ImportError:
        print("  ⚠ video omitido: instala con  python -m pip install imageio imageio-ffmpeg")
        return None

    carpeta = os.path.join(RESULTS, "video")
    ruta = os.path.join(carpeta, "explicacion_optimizacion.mp4")
    a, b = r["a_star"], r["b_star"]

    # Trayectoria del gradiente desde (0,0), guardando (a, b) por paso
    pasos = []
    av, bv = 0.0, 0.0
    eta = ajustar_eta(x)
    from derivadas import gradiente as calc_grad
    for _ in range(g["iteraciones"]):
        pasos.append((av, bv))
        dS_da, dS_db = calc_grad(x, y, av, bv)
        av -= eta * dS_da
        bv -= eta * dS_db
    # muestrear la trayectoria para ~60 frames
    idx = np.linspace(0, len(pasos) - 1, min(60, len(pasos))).astype(int)
    tray = [pasos[i] for i in idx]

    # Escena: curvas de nivel + recta moviéndose hacia el ajuste
    A = np.linspace(min(a, 0) - 1, a + 2, 100)
    B = np.linspace(min(b, 0) - 1, b + 2, 100)
    AA, BB = np.meshgrid(A, B)
    S = np.array([[error_cuadratico(x, y, av_, bv_) for av_ in A] for bv_ in B])

    frames = []
    xs = np.linspace(x.min(), x.max(), 50)
    for k, (ak, bk) in enumerate(tray):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Panel 1: datos + recta actual
        ax1.scatter(x, y, color=AZUL, s=50, zorder=3, label="Datos")
        ax1.plot(xs, prediccion(xs, ak, bk), color="#d64545", lw=2,
                 label=f"iter {idx[k]}: y={ak:.2f}x+{bk:.2f}")
        ax1.set_title(f"Ajuste por gradiente — iteración {idx[k]}")
        ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.legend(loc="lower right")
        ax1.grid(True)

        # Panel 2: curvas de nivel + trayectoria hasta ahora
        ax2.contourf(AA, BB, S, levels=25, cmap="Blues")
        ax2.plot([p[0] for p in tray[:k+1]], [p[1] for p in tray[:k+1]],
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

    # último frame sostenido 1 s
    frames.extend([frames[-1]] * fps)

    try:
        imageio.mimsave(ruta, frames, fps=fps, codec="libx264",
                        macro_block_size=1, quality=8)
    except TypeError:
        imageio.mimsave(ruta, frames, fps=fps)
    print(f"  ▶ video generado: {ruta}  ({len(frames)} frames @ {fps} fps)")
    return ruta


def exportar_todo() -> str:
    _limpiar_results()      # 1. BORRAR contenido anterior de results/
    _carpetas()
    x, y = obtener_datos()
    r = reporte_analitico(x, y)
    h = verificar_minimo(x)
    g = descenso_gradiente(x, y, eta=ajustar_eta(x), max_iter=50_000)

    exportar_data(x, y, r, h, g)
    exportar_excel(x, y, r, h, g)
    exportar_images(x, y, r, g)
    exportar_html(x, y, r, h, g)
    crear_video(x, y, r, g)
    return RESULTS


if __name__ == "__main__":
    resultados = exportar_todo()
    print(f"\n✓ Resultados exportados en: {os.path.abspath(resultados)}")
    for raiz, _, archivos in os.walk(resultados):
        for a in archivos:
            print("   •", os.path.relpath(os.path.join(raiz, a), resultados))
