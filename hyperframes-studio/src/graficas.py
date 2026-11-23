"""
graficas.py — figuras del proyecto (results/images/)
-----------------------------------------------------
Cada figura corresponde a una sección del enunciado:

  01_dispersion.png          §5    diagrama de dispersión (exploración)
  02_regresion.png           §8.4  datos + recta óptima + otras rectas (a, b)
  03_superficie_error.png    §7.5–7.6  superficie z = S(a, b) (paraboloide)
  04_trazas.png              §7.7  trazas y secciones transversales
  05_curvas_nivel.png        §7.8  curvas de nivel + (a*, b*)
  06_residuos_vs_x.png       §11   residuos e_i en función de x_i
  07_histograma_residuos.png §11   histograma de residuos
  08_metodos_numericos.png   extra trayectorias de gradiente / Newton
  09_convergencia.png        extra S(k) por iteración
  10_influencia.png          §13-B influencia de cada dato
  11_lineal_vs_cuadratico.png §13-C comparación con y = ax² + bx + c

IMPORTANTE (interpretación geométrica): S(a, b) es un polinomio de grado 2
en (a, b) con Hessiano constante y definido positivo, así que su gráfica es
SIEMPRE un PARABOLOIDE ELÍPTICO (un "tazón") con UN solo mínimo. No tiene
ondulaciones ni varios mínimos.
"""

import os
import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm

from modelo import prediccion, error_cuadratico
from derivadas import hessiana
from datos import ETIQUETA_X, ETIQUETA_Y, UNIDAD_X, UNIDAD_Y, PROBLEMA

AZUL, AZUL_OSC, NARANJA, ROJO, VERDE, GRIS = (
    "#0B5FCC", "#083A80", "#E8740C", "#D62728", "#1B8A4C", "#5A6472")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRIS, "axes.labelcolor": AZUL_OSC, "axes.titlecolor": AZUL_OSC,
    "axes.grid": True, "grid.color": "#C9D6E8", "grid.linewidth": 0.6, "grid.alpha": 0.6,
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12.5,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110,
})

CARPETA_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "images")
os.makedirs(CARPETA_SALIDA, exist_ok=True)


def _guardar(fig, nombre):
    fig.savefig(os.path.join(CARPETA_SALIDA, nombre), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _malla_S(x, y, a_vals, b_vals):
    """S[i, j] = S(a_vals[j], b_vals[i]) vectorizado."""
    A, B = np.meshgrid(a_vals, b_vals)
    x = np.asarray(x, float); y = np.asarray(y, float)
    S = ((y[None, None, :] - A[..., None] * x[None, None, :] - B[..., None]) ** 2).sum(-1)
    return A, B, S


def ventana(x, a_star, b_star, delta):
    """
    Intervalo ADECUADO de (a, b): la curva de nivel S = S_mín + Δ es la elipse
    ½·pᵀHp = Δ, cuyos extremos en cada eje son  ±√(2Δ·(H⁻¹)_aa)  y  ±√(2Δ·(H⁻¹)_bb).
    """
    Hinv = np.linalg.inv(hessiana(x))
    da = np.sqrt(2 * delta * Hinv[0, 0])
    db = np.sqrt(2 * delta * Hinv[1, 1])
    return (a_star - da, a_star + da), (b_star - db, b_star + db)


# ═════════════════════════ 01 · DISPERSIÓN ═════════════════════════
def grafica_dispersion(x, y):
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.scatter(x, y, s=75, color=AZUL_OSC, edgecolor="white", lw=1.4, zorder=3)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.0f}", (xi, yi), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=GRIS)
    r = np.corrcoef(x, y)[0, 1]
    ax.set_title(f"Diagrama de dispersión — {PROBLEMA}\n"
                 f"n = {len(x)}, coeficiente de correlación r = {r:.4f}")
    ax.set_xlabel(ETIQUETA_X); ax.set_ylabel(ETIQUETA_Y)
    _guardar(fig, "01_dispersion.png")


# ═════════════════════ 02 · RECTA ÓPTIMA vs OTRAS ═════════════════════
def grafica_regresion(x, y, a, b):
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    xs = np.linspace(x.min() - 1, x.max() + 1, 200)
    x_mean, y_mean = x.mean(), y.mean()
    # otras rectas: misma pendiente desplazada, pendiente distinta por (x̄, ȳ),
    # y una recta "a ojo" por los extremos
    a_ext = (y[-1] - y[0]) / (x[-1] - x[0]); b_ext = y[0] - a_ext * x[0]
    otras = [
        (3.0, y_mean - 3.0 * x_mean, "pendiente 3.0 por (x̄, ȳ)", NARANJA, "--"),
        (4.0, y_mean - 4.0 * x_mean, "pendiente 4.0 por (x̄, ȳ)", VERDE, "--"),
        (a, b + 3.0, "recta óptima desplazada +3 lb", GRIS, ":"),
        (a_ext, b_ext, "recta por el primer y el último punto", "#8E44AD", "-."),
    ]
    for ai, bi, nom, col, ls in otras:
        ax.plot(xs, prediccion(xs, ai, bi), color=col, ls=ls, lw=1.6,
                label=f"{nom}: a={ai:.3f}, b={bi:.2f} → S={error_cuadratico(x, y, ai, bi):.1f}")
    ax.plot(xs, prediccion(xs, a, b), color=AZUL, lw=3,
            label=f"ÓPTIMA  ŷ = {a:.4f}x {b:+.4f} → S = {error_cuadratico(x, y, a, b):.2f} (mínimo)")
    ax.scatter(x, y, s=70, color=AZUL_OSC, edgecolor="white", lw=1.4, zorder=5,
               label="Datos observados")
    ax.set_title("Recta de mínimos cuadrados vs otras rectas: la óptima tiene el menor S(a, b)")
    ax.set_xlabel(ETIQUETA_X); ax.set_ylabel(ETIQUETA_Y)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _guardar(fig, "02_regresion.png")


# ═════════════════════ 03 · SUPERFICIE ═════════════════════
def grafica_superficie_error(x, y, a_star, b_star):
    s_min = error_cuadratico(x, y, a_star, b_star)
    fig = plt.figure(figsize=(15, 6.5))

    # (izq.) variables originales (a, b): valle largo y estrecho
    # Malla ELÍPTICA en la base de autovectores de H: cubre exactamente la
    # región S ≤ S_mín + Δ (Δ = 6·S_mín) y da bordes suaves.
    lam, V = np.linalg.eigh(hessiana(x))
    delta = 6 * s_min
    r = np.linspace(0, 1, 60)[:, None]; th = np.linspace(0, 2 * np.pi, 121)[None, :]
    s1 = np.sqrt(2 * delta / lam[0]); s2 = np.sqrt(2 * delta / lam[1])
    A = a_star + r * (np.cos(th) * s1 * V[0, 0] + np.sin(th) * s2 * V[0, 1])
    B = b_star + r * (np.cos(th) * s1 * V[1, 0] + np.sin(th) * s2 * V[1, 1])
    S = ((y[None, None, :] - A[..., None] * x[None, None, :] - B[..., None]) ** 2).sum(-1)
    ax = fig.add_subplot(121, projection="3d")
    ax.plot_surface(A, B, S, cmap="viridis", rstride=1, cstride=1, lw=0,
                    antialiased=True, alpha=0.95, vmin=s_min, vmax=7 * s_min)
    for k_ in (0.25, 0.5, 0.75, 1.0):                 # curvas de nivel proyectadas (elipses)
        ax.plot(A[int(k_ * 59)], B[int(k_ * 59)], np.zeros(121), color=GRIS, lw=0.8)
    ax.scatter([a_star], [b_star], [s_min], color=ROJO, s=140, marker="*", depthshade=False,
               label=f"mínimo (a*, b*) = ({a_star:.2f}, {b_star:.2f}), S = {s_min:.2f}")
    ax.set_title("z = S(a, b): paraboloide con un valle diagonal\n(a y b están fuertemente acoplados)")
    ax.set_xlabel("a (pendiente) [lb/in]"); ax.set_ylabel("b (intercepto) [lb]"); ax.set_zlabel("S [lb²]")
    ax.view_init(elev=32, azim=-60)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    # (der.) mismo S con x centrada: y = a(x − x̄) + c,  c = b + a·x̄
    x_mean = x.mean(); c_star = b_star + a_star * x_mean
    u = x - x_mean
    da = np.sqrt(2 * 6 * s_min / (2 * np.sum(u ** 2)))
    dc = np.sqrt(2 * 6 * s_min / (2 * len(x)))
    a_v = np.linspace(a_star - da, a_star + da, 160)
    c_v = np.linspace(c_star - dc, c_star + dc, 160)
    A2, C2, S2 = _malla_S(u, y, a_v, c_v)
    S2 = np.where((A2 - a_star) ** 2 / da ** 2 + (C2 - c_star) ** 2 / dc ** 2 <= 1, S2, np.nan)
    ax = fig.add_subplot(122, projection="3d")
    ax.plot_surface(A2, C2, S2, cmap="viridis", rstride=2, cstride=2, lw=0.1,
                    edgecolor="#083A8022", alpha=0.95, vmin=s_min, vmax=7 * s_min)
    ax.contour(A2, C2, S2, levels=8, zdir="z", offset=0, cmap="viridis", linewidths=0.8)
    ax.scatter([a_star], [c_star], [s_min], color=ROJO, s=140, marker="*", depthshade=False)
    ax.set_title("La misma función con x centrada (c = b + a·x̄):\nun tazón — paraboloide elíptico")
    ax.set_xlabel("a [lb/in]"); ax.set_ylabel("c = ŷ(x̄) [lb]"); ax.set_zlabel("S [lb²]")
    ax.view_init(elev=28, azim=-55)
    fig.tight_layout()
    _guardar(fig, "03_superficie_error.png")


# ═════════════════════ 04 · TRAZAS ═════════════════════
def grafica_trazas(x, y, a_star, b_star):
    s_min = error_cuadratico(x, y, a_star, b_star)
    H = hessiana(x)
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))

    # (a) plano b = constante → S(a, b_k) es una parábola en a
    ax = axs[0]
    a_v = np.linspace(a_star - 0.12, a_star + 0.12, 300)
    for db, col in [(-2, NARANJA), (0, AZUL), (2, VERDE)]:
        bk = b_star + db
        Sv = [error_cuadratico(x, y, av, bk) for av in a_v]
        amin = a_v[int(np.argmin(Sv))]
        ax.plot(a_v, Sv, color=col, lw=2.4 if db == 0 else 1.6,
                label=f"b = {bk:.2f}  (mín. en a ≈ {amin:.4f})")
    ax.axvline(a_star, color=ROJO, ls=":", lw=1)
    ax.set_title("Trazas en planos b = cte:  S(a, b₀)\nparábolas que abren hacia arriba")
    ax.set_xlabel("a (pendiente) [lb/in]"); ax.set_ylabel("S [lb²]")
    ax.legend(frameon=False, fontsize=9, title=f"∂²S/∂a² = 2Σx² = {H[0,0]:.0f}",
              title_fontsize=9)

    # (b) plano a = constante → S(a_k, b) es una parábola en b
    ax = axs[1]
    b_v = np.linspace(b_star - 8, b_star + 8, 300)
    for da, col in [(-0.02, NARANJA), (0, AZUL), (0.02, VERDE)]:
        ak = a_star + da
        Sv = [error_cuadratico(x, y, ak, bv) for bv in b_v]
        bmin = b_v[int(np.argmin(Sv))]
        ax.plot(b_v, Sv, color=col, lw=2.4 if da == 0 else 1.6,
                label=f"a = {ak:.2f}  (mín. en b ≈ {bmin:.2f})")
    ax.axvline(b_star, color=ROJO, ls=":", lw=1)
    ax.set_title("Trazas en planos a = cte:  S(a₀, b)\nparábolas mucho más 'planas'")
    ax.set_xlabel("b (intercepto) [lb]"); ax.set_ylabel("S [lb²]")
    ax.legend(frameon=False, fontsize=9, title=f"∂²S/∂b² = 2n = {H[1,1]:.0f}",
              title_fontsize=9)

    # (c) sección transversal a lo largo de los ejes principales (autovectores de H)
    ax = axs[2]
    lam, V = np.linalg.eigh(H)
    t = np.linspace(-1, 1, 300)
    for k, (col, nom) in enumerate([(VERDE, "dirección PLANA (λ₁)"), (NARANJA, "dirección EMPINADA (λ₂)")]):
        v = V[:, k]
        escala = np.sqrt(2 * 3 * s_min / lam[k])      # sube hasta S_mín + 3·S_mín
        Sv = [error_cuadratico(x, y, a_star + ti * escala * v[0], b_star + ti * escala * v[1]) for ti in t]
        ax.plot(t, Sv, color=col, lw=2.6 if k == 0 else 1.6, ls="-" if k == 0 else "--",
                label=f"{nom}: v = ({v[0]:+.4f}, {v[1]:+.4f}), λ = {lam[k]:.3g}\n"
                      f"   S llega a 4·S_mín a distancia {escala:.4g}")
    ax.set_title("Secciones transversales por (a*, b*) en las\ndirecciones de los autovectores de H (eje x normalizado)")
    ax.set_xlabel("t = distancia / (distancia donde S = 4·S_mín)"); ax.set_ylabel("S [lb²]")
    ax.set_ylim(0, 4.6 * s_min)
    ax.legend(frameon=False, fontsize=8.2, loc="upper center")
    ax.text(0.5, 0.03, "mismas parábolas, pero la plana es ~1000 veces más ancha",
            transform=ax.transAxes, ha="center", fontsize=9, color=GRIS)
    fig.tight_layout()
    _guardar(fig, "04_trazas.png")


# ═════════════════════ 05 · CURVAS DE NIVEL ═════════════════════
def grafica_curvas_nivel(x, y, a_star, b_star):
    s_min = error_cuadratico(x, y, a_star, b_star)
    k_rel = np.array([0.1, 0.25, 0.5, 1, 2, 4, 8])
    niveles = s_min * (1 + k_rel)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8))

    # (izq.) variables originales (a, b)
    (a0, a1), (b0, b1) = ventana(x, a_star, b_star, 9 * s_min)
    A, B, S = _malla_S(x, y, np.linspace(a0, a1, 500), np.linspace(b0, b1, 500))
    cf = ax1.contourf(A, B, S, levels=np.logspace(np.log10(s_min), np.log10(S.max()), 40),
                      norm=LogNorm(), cmap="viridis_r")
    cs = ax1.contour(A, B, S, levels=niveles, colors="white", linewidths=1.0)
    ax1.clabel(cs, cs.levels[3:], fmt=lambda v: f"{v:.0f}", fontsize=8)
    ax1.scatter([a_star], [b_star], color=ROJO, s=260, marker="*", zorder=5, edgecolor="white",
                label=f"(a*, b*) = ({a_star:.4f}, {b_star:.4f})\nS_mín = {s_min:.4f} lb²")
    fig.colorbar(cf, ax=ax1, label="S(a, b) [lb²] (escala log)", format="%.0f")
    ax1.set_title("Curvas de nivel S(a, b) = k en el plano (a, b)\n"
                  "elipses concéntricas MUY alargadas (κ ≈ 10⁶) e inclinadas")
    ax1.set_xlabel("a (pendiente) [lb/in]"); ax1.set_ylabel("b (intercepto) [lb]")
    ax1.legend(frameon=True, fontsize=9, loc="upper right")

    # (der.) mismas curvas en (a, c), c = b + a·x̄  (x centrada)
    x_mean = x.mean(); c_star = b_star + a_star * x_mean; u = x - x_mean
    da = np.sqrt(2 * 9 * s_min / (2 * np.sum(u ** 2))); dc = np.sqrt(2 * 9 * s_min / (2 * len(x)))
    A2, C2, S2 = _malla_S(u, y, np.linspace(a_star - da, a_star + da, 400),
                          np.linspace(c_star - dc, c_star + dc, 400))
    cf2 = ax2.contourf(A2, C2, S2, levels=30, cmap="viridis_r")
    cs2 = ax2.contour(A2, C2, S2, levels=niveles, colors="white", linewidths=1.0)
    ax2.clabel(cs2, fmt=lambda v: f"{v:.0f}", fontsize=8)
    ax2.scatter([a_star], [c_star], color=ROJO, s=260, marker="*", zorder=5, edgecolor="white",
                label=f"(a*, c*) = ({a_star:.4f}, {c_star:.4f})")
    fig.colorbar(cf2, ax=ax2, label="S [lb²]")
    ax2.set_title("Las mismas curvas con x centrada: c = b + a·x̄\n"
                  "elipses alineadas con los ejes (H diagonal)")
    ax2.set_xlabel("a (pendiente) [lb/in]"); ax2.set_ylabel("c = ŷ(x̄) [lb]")
    ax2.legend(frameon=True, fontsize=9, loc="upper right")
    fig.tight_layout()
    _guardar(fig, "05_curvas_nivel.png")


# ═════════════════════ 06–07 · RESIDUOS ═════════════════════
def grafica_residuos_vs_x(x, y, a, b, res):
    e = res["e"]; s = res["s"]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.axhline(0, color=GRIS, lw=1.2)
    ax.axhspan(-2 * s, 2 * s, color=AZUL, alpha=0.07, label=f"±2s  (s = {s:.3f} {UNIDAD_Y})")
    ax.vlines(x, 0, e, colors=[VERDE if v > 0 else NARANJA for v in e], lw=2)
    ax.scatter(x, e, s=70, color=[VERDE if v > 0 else NARANJA for v in e],
               edgecolor="white", zorder=4)
    # tendencia de los residuos (parábola) para visualizar el patrón
    c = np.polyfit(x, e, 2); xs = np.linspace(x.min(), x.max(), 200)
    ax.plot(xs, np.polyval(c, xs), color=ROJO, ls="--", lw=1.6,
            label="tendencia de los residuos (patrón en U)")
    ax.set_title("Residuos eᵢ = yᵢ − (a*xᵢ + b*) en función de xᵢ\n"
                 "positivos en los extremos, negativos en el centro → desviación sistemática")
    ax.set_xlabel(ETIQUETA_X); ax.set_ylabel(f"residuo eᵢ [{UNIDAD_Y}]")
    ax.legend(frameon=False, fontsize=9)
    _guardar(fig, "06_residuos_vs_x.png")


def grafica_histograma_residuos(res):
    e = res["e"]; s = res["s"]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    bins = np.arange(np.floor(e.min()) - 0.5, np.ceil(e.max()) + 1.0, 0.5)
    ax.hist(e, bins=bins, color=AZUL, edgecolor="white", alpha=0.85, label="frecuencia")
    xs = np.linspace(e.min() - 1, e.max() + 1, 200)
    ax.plot(xs, len(e) * 0.5 * np.exp(-xs ** 2 / (2 * s ** 2)) / (s * np.sqrt(2 * np.pi)),
            color=ROJO, lw=2, label=f"normal N(0, s²) de referencia")
    ax.axvline(0, color=GRIS, lw=1)
    ax.axvline(np.mean(e), color=NARANJA, ls="--", lw=1.6, label=f"media = {np.mean(e):.1e}")
    ax.set_title("Histograma de los residuos\n"
                 "centrado en 0, pero asimétrico: muchos pequeños negativos y pocos grandes positivos")
    ax.set_xlabel(f"residuo eᵢ [{UNIDAD_Y}]"); ax.set_ylabel("número de datos")
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(frameon=False, fontsize=9)
    _guardar(fig, "07_histograma_residuos.png")


# ═════════════════════ 08–09 · MÉTODOS NUMÉRICOS ═════════════════════
def grafica_metodos_numericos(x, y, a_star, b_star, gd_simple, gd_centrado, newton):
    fig, ax = plt.subplots(figsize=(9.5, 7))
    a_v = np.linspace(-0.6, 4.3, 300); b_v = np.linspace(-225, 15, 300)
    A, B, S = _malla_S(x, y, a_v, b_v)
    cf = ax.contourf(A, B, S, levels=np.logspace(np.log10(S.min()), np.log10(S.max()), 30),
                     norm=LogNorm(), cmap="Blues_r")
    fig.colorbar(cf, ax=ax, label="S(a, b) [lb²] (escala log)")
    for res, col, nom in [(gd_simple, NARANJA, "Gradiente simple"),
                          (gd_centrado, VERDE, "Gradiente con x centrada")]:
        tr = res["trayectoria"]
        ax.plot([p[0] for p in tr], [p[1] for p in tr], "-o", color=col, ms=2.5, lw=1.6,
                label=f"{nom}: {res['iteraciones']} iter., "
                      f"{'convergió' if res['convergio'] else 'NO convergió'} → "
                      f"({res['a_star']:.3f}, {res['b_star']:.2f})")
    ax.plot([0, newton["a_star"]], [0, newton["b_star"]], "--", color=ROJO, lw=1.6,
            label="Newton: 1 paso (a,b) − H⁻¹∇S → mínimo exacto")
    ax.scatter([0], [0], color="black", s=60, zorder=6, label="inicio (0, 0)")
    ax.scatter([a_star], [b_star], color=ROJO, s=280, marker="*", zorder=6, edgecolor="white",
               label=f"(a*, b*) = ({a_star:.3f}, {b_star:.2f})")
    ax.set_title("Métodos numéricos sobre las curvas de nivel de S(a, b)\n"
                 "el gradiente simple cae al valle y avanza muy lento por él")
    ax.set_xlabel("a (pendiente) [lb/in]"); ax.set_ylabel("b (intercepto) [lb]")
    ax.legend(frameon=True, fontsize=8, loc="lower left")
    _guardar(fig, "08_metodos_numericos.png")


def grafica_convergencia(x, y, gd_simple, gd_centrado, s_min):
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for res, col, nom in [(gd_simple, NARANJA, "gradiente simple"),
                          (gd_centrado, VERDE, "gradiente con x centrada")]:
        k = np.array([p[4] for p in res["trayectoria"]], float)
        Sv = np.array([p[2] for p in res["trayectoria"]]) - s_min
        ax.plot(np.maximum(k, 1), np.maximum(Sv, 1e-12), color=col, lw=2, label=nom)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title("Convergencia: S(a_k, b_k) − S_mín por iteración")
    ax.set_xlabel("iteración k (escala log)"); ax.set_ylabel("S − S_mín [lb²] (escala log)")
    ax.legend(frameon=False)
    _guardar(fig, "09_convergencia.png")


# ═════════════════════ 10 · INFLUENCIA ═════════════════════
def grafica_influencia(x, y, inf):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.4))
    xs = np.linspace(x.min() - 1, x.max() + 1, 100)
    for f in inf["filas"]:
        ax1.plot(xs, prediccion(xs, f["a"], f["b"]), color=GRIS, lw=0.8, alpha=0.6)
    m = inf["mas_influyente"]
    ax1.plot(xs, prediccion(xs, m["a"], m["b"]), color=NARANJA, lw=2,
             label=f"sin el punto x={m['x']:.0f} (más influyente)")
    ax1.plot(xs, prediccion(xs, inf["a0"], inf["b0"]), color=AZUL, lw=2.6, label="con todos los datos")
    ax1.scatter(x, y, color=AZUL_OSC, s=55, zorder=4, edgecolor="white")
    ax1.scatter([m["x"]], [m["y"]], s=200, facecolor="none", edgecolor=NARANJA, lw=2, zorder=5)
    ax1.set_title("§13-B: recta recalculada quitando un punto a la vez (líneas grises)")
    ax1.set_xlabel(ETIQUETA_X); ax1.set_ylabel(ETIQUETA_Y); ax1.legend(frameon=False, fontsize=9)

    xi = [f["x"] for f in inf["filas"]]
    ax2.bar(xi, [f["da"] for f in inf["filas"]],
            color=[NARANJA if f is m else AZUL for f in inf["filas"]])
    ax2.axhline(0, color=GRIS, lw=1)
    ax2.set_title("Cambio de la pendiente Δa al eliminar cada punto")
    ax2.set_xlabel(f"x del punto eliminado [{UNIDAD_X}]"); ax2.set_ylabel(f"Δa [{UNIDAD_Y}/{UNIDAD_X}]")
    fig.tight_layout()
    _guardar(fig, "10_influencia.png")


# ═════════════════════ 11 · LINEAL vs CUADRÁTICO ═════════════════════
def grafica_lineal_vs_cuadratico(x, y, cq):
    a, b = cq["lineal"]; p, q, r = cq["cuadratico"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.4))
    xs = np.linspace(x.min() - 1, x.max() + 1, 200)
    ax1.scatter(x, y, color=AZUL_OSC, s=60, zorder=4, edgecolor="white", label="datos")
    ax1.plot(xs, a * xs + b, color=AZUL, lw=2.2, label=f"lineal: S = {cq['S_lineal']:.2f}, R² = {cq['R2_lineal']:.4f}")
    ax1.plot(xs, p * xs ** 2 + q * xs + r, color=ROJO, lw=2, ls="--",
             label=f"cuadrático: S = {cq['S_cuad']:.2f}, R² = {cq['R2_cuad']:.4f}")
    ax1.set_title("§13-C: modelo lineal vs y = p·x² + q·x + r")
    ax1.set_xlabel(ETIQUETA_X); ax1.set_ylabel(ETIQUETA_Y); ax1.legend(frameon=False, fontsize=9)

    el = y - (a * x + b)
    ax2.axhline(0, color=GRIS, lw=1)
    ax2.plot(x, el, "o-", color=AZUL, label="residuos lineal (patrón en U)")
    ax2.plot(x, cq["e_cuad"], "s-", color=ROJO, label="residuos cuadrático (sin patrón claro)")
    ax2.set_title("Residuos de ambos modelos")
    ax2.set_xlabel(ETIQUETA_X); ax2.set_ylabel(f"residuo [{UNIDAD_Y}]"); ax2.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _guardar(fig, "11_lineal_vs_cuadratico.png")


# ═══════════════════════ ORQUESTADOR ═══════════════════════
def generar_todas(x, y, a_star, b_star, res, gd_simple, gd_centrado, newton, inf, cq):
    print("\nGenerando gráficas...")
    s_min = error_cuadratico(x, y, a_star, b_star)
    grafica_dispersion(x, y)
    grafica_regresion(x, y, a_star, b_star)
    grafica_superficie_error(x, y, a_star, b_star)
    grafica_trazas(x, y, a_star, b_star)
    grafica_curvas_nivel(x, y, a_star, b_star)
    grafica_residuos_vs_x(x, y, a_star, b_star, res)
    grafica_histograma_residuos(res)
    grafica_metodos_numericos(x, y, a_star, b_star, gd_simple, gd_centrado, newton)
    grafica_convergencia(x, y, gd_simple, gd_centrado, s_min)
    grafica_influencia(x, y, inf)
    grafica_lineal_vs_cuadratico(x, y, cq)
    print("OK - 11 gráficas guardadas en:", os.path.abspath(CARPETA_SALIDA))
