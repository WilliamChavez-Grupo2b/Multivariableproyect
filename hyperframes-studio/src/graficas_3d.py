"""
graficas_3d.py — gráficas 3D estáticas (PNG) en results/images/
----------------------------------------------------------------
Versión en imagen de los modelos 3D (se ven en GitHub y sirven para las
diapositivas). La versión interactiva está en results/html/modelos_3d.html.

  12_3d_superficie_trazas.png   superficie S(a, b), trazas, planos de corte, mínimo
  13_3d_curvas_nivel.png        curvas de nivel sobre la superficie y proyectadas
  14_3d_tazon_centrado.png      la misma S con x centrada: paraboloide elíptico
  15_3d_recta_y_superficie.png  tres rectas y su punto sobre la superficie
  16_3d_metodos_numericos.png   gradiente simple, gradiente centrado y Newton en 3D
  17_3d_contribucion_errores.png  barras 3D: aporte eᵢ² de cada dato a S
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from modelo import error_cuadratico
from derivadas import hessiana
from datos import ETIQUETA_X, ETIQUETA_Y

AZUL, AZUL_OSC, NARANJA, ROJO, VERDE, GRIS = (
    "#0B5FCC", "#083A80", "#E8740C", "#D62728", "#1B8A4C", "#5A6472")
CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "images")


def _S(x, y, A, B):
    return ((y[None, None, :] - A[..., None] * x[None, None, :] - B[..., None]) ** 2).sum(-1)


def _guardar(fig, nombre):
    os.makedirs(CARPETA, exist_ok=True)
    fig.savefig(os.path.join(CARPETA, nombre), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _ejes(ax, xl, yl, zl="S(a, b) [lb²]"):
    ax.set_xlabel(xl, labelpad=8); ax.set_ylabel(yl, labelpad=8); ax.set_zlabel(zl, labelpad=6)
    for eje in (ax.xaxis, ax.yaxis, ax.zaxis):
        eje.pane.set_alpha(0.15)


# ═════════════════ 12 · SUPERFICIE + TRAZAS ═════════════════
def superficie_trazas(x, y, a_s, b_s, s_min, da=0.08, db=5.0):
    a_v = np.linspace(a_s - da, a_s + da, 80); b_v = np.linspace(b_s - db, b_s + db, 80)
    A, B = np.meshgrid(a_v, b_v); S = _S(x, y, A, B)
    zt = S.max()
    fig = plt.figure(figsize=(12, 9)); ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    ax.plot_surface(A, B, S, cmap="viridis", alpha=0.72, rstride=2, cstride=2, lw=0.2,
                    edgecolor="#ffffff55")
    # planos de corte
    PA, PZ = np.meshgrid(a_v[[0, -1]], [0, zt])
    ax.plot_surface(PA, np.full_like(PA, b_s), PZ, color=NARANJA, alpha=0.12)
    PB, PZ2 = np.meshgrid(b_v[[0, -1]], [0, zt])
    ax.plot_surface(np.full_like(PB, a_s), PB, PZ2, color=AZUL, alpha=0.10)
    # trazas
    ax.plot(a_v, np.full_like(a_v, b_s), [error_cuadratico(x, y, a, b_s) for a in a_v],
            color=NARANJA, lw=4, label="traza con el plano b = b*  →  parábola en a")
    ax.plot(np.full_like(b_v, a_s), b_v, [error_cuadratico(x, y, a_s, b) for b in b_v],
            color=AZUL, lw=4, label="traza con el plano a = a*  →  parábola en b")
    # fondo del valle
    bv = y.mean() - a_v * x.mean()
    ax.plot(a_v, bv, [error_cuadratico(x, y, a, b) + 3 for a, b in zip(a_v, bv)], color="black",
            lw=2.2, ls="--", label="fondo del valle  b = ȳ − a·x̄")
    ax.scatter([a_s], [b_s], [s_min], color=ROJO, s=160, marker="D", depthshade=False, zorder=10,
               label=f"mínimo (a*, b*) = ({a_s:.4f}, {b_s:.4f}),  S = {s_min:.2f}")
    ax.contour(A, B, S, levels=10, zdir="z", offset=0, cmap="viridis", linewidths=1)
    _ejes(ax, "a (pendiente) [lb/in]", "b (intercepto) [lb]")
    ax.set_zlim(0, zt); ax.view_init(elev=22, azim=128)
    ax.set_title("Superficie de error z = S(a, b) con sus trazas\n"
                 "paraboloide: las trazas son parábolas que abren hacia arriba → mínimo",
                 fontsize=13, fontweight="bold", color=AZUL_OSC)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    _guardar(fig, "12_3d_superficie_trazas.png")


# ═════════════════ 13 · CURVAS DE NIVEL EN 3D ═════════════════
def curvas_nivel_3d(x, y, a_s, b_s, s_min):
    xm = x.mean(); u = x - xm; c_s = b_s + a_s * xm
    da = np.sqrt(8 * s_min / (2 * np.sum(u ** 2))); dc = np.sqrt(8 * s_min / (2 * len(x)))
    a_v = np.linspace(a_s - da, a_s + da, 70); c_v = np.linspace(c_s - dc, c_s + dc, 70)
    A, C = np.meshgrid(a_v, c_v); S = _S(u, y, A, C)
    niveles = s_min * np.array([1.25, 1.75, 2.5, 3.5, 4.5])
    fig = plt.figure(figsize=(15, 6.8))

    ax = fig.add_subplot(121, projection="3d")
    ax.plot_surface(A, C, S, cmap="viridis", alpha=0.35, lw=0)
    ax.contour(A, C, S, levels=niveles, cmap="plasma", linewidths=2.5)            # sobre la superficie
    ax.contour(A, C, S, levels=niveles, zdir="z", offset=0, cmap="plasma", linewidths=1.5)  # proyección
    ax.scatter([a_s], [c_s], [s_min], color=ROJO, s=120, marker="D", depthshade=False)
    ax.scatter([a_s], [c_s], [0], color=ROJO, s=60, marker="x", depthshade=False)
    ax.plot([a_s, a_s], [c_s, c_s], [0, s_min], color=ROJO, ls=":", lw=1.5)
    _ejes(ax, "a [lb/in]", "c = b + a·x̄ [lb]"); ax.set_zlim(0, S.max())
    ax.view_init(elev=22, azim=-60)
    ax.set_title("Curvas de nivel S = k: cortes horizontales\nde la superficie y su proyección en el piso",
                 fontsize=12, fontweight="bold", color=AZUL_OSC)

    ax2 = fig.add_subplot(122)
    cs = ax2.contour(A, C, S, levels=niveles, cmap="plasma", linewidths=2)
    ax2.clabel(cs, fmt=lambda v: f"S={v:.0f}", fontsize=9)
    ax2.scatter([a_s], [c_s], color=ROJO, s=200, marker="*", zorder=5,
                label=f"(a*, c*) = ({a_s:.3f}, {c_s:.2f})\n→ b* = c* − a*·x̄ = {b_s:.3f}")
    ax2.set_xlabel("a [lb/in]"); ax2.set_ylabel("c = b + a·x̄ [lb]"); ax2.grid(alpha=0.4)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_title("Vista desde arriba: elipses concéntricas\ncentradas en el mínimo",
                  fontsize=12, fontweight="bold", color=AZUL_OSC)
    fig.tight_layout()
    _guardar(fig, "13_3d_curvas_nivel.png")


# ═════════════════ 14 · TAZÓN CENTRADO ═════════════════
def tazon_centrado(x, y, a_s, b_s, s_min):
    xm = x.mean(); u = x - xm; c_s = b_s + a_s * xm
    da = np.sqrt(12 * s_min / (2 * np.sum(u ** 2))); dc = np.sqrt(12 * s_min / (2 * len(x)))
    r = np.linspace(0, 1, 50)[:, None]; th = np.linspace(0, 2 * np.pi, 120)[None, :]
    A = a_s + r * da * np.cos(th); C = c_s + r * dc * np.sin(th); S = _S(u, y, A, C)
    fig = plt.figure(figsize=(11, 8.5)); ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    ax.plot_surface(A, C, S, cmap="viridis", alpha=0.9, rstride=2, cstride=3, lw=0.2,
                    edgecolor="#ffffff40")
    ax.contour(A, C, S, levels=8, zdir="z", offset=0, cmap="viridis", linewidths=1.2)
    ax.scatter([a_s], [c_s], [s_min], color=ROJO, s=170, marker="D", depthshade=False, zorder=10,
               label=f"mínimo único: S = {s_min:.2f} lb²")
    ax.plot([a_s, a_s], [c_s, c_s], [0, s_min], color=ROJO, ls=":", lw=1.5)
    _ejes(ax, "a (pendiente) [lb/in]", "c = b + a·x̄ [lb]"); ax.set_zlim(0, S.max())
    ax.view_init(elev=26, azim=-55)
    H = 2 * np.array([[np.sum(u ** 2), 0], [0, len(x)]])
    ax.set_title("Paraboloide elíptico (x centrada): un “tazón” con un solo mínimo\n"
                 f"Hessiano diagonal H = diag({H[0,0]:.0f}, {H[1,1]:.0f}) > 0",
                 fontsize=13, fontweight="bold", color=AZUL_OSC)
    ax.legend(loc="upper left")
    _guardar(fig, "14_3d_tazon_centrado.png")


# ═════════════════ 15 · RECTA ↔ PUNTO SOBRE LA SUPERFICIE ═════════════════
def recta_y_superficie(x, y, a_s, b_s, s_min):
    xm = x.mean(); u = x - xm; c_s = b_s + a_s * xm
    da = np.sqrt(12 * s_min / (2 * np.sum(u ** 2))); dc = np.sqrt(12 * s_min / (2 * len(x)))
    r = np.linspace(0, 1, 40)[:, None]; th = np.linspace(0, 2 * np.pi, 100)[None, :]
    A = a_s + r * da * np.cos(th); C = c_s + r * dc * np.sin(th); S = _S(u, y, A, C)
    casos = [(a_s + 0.85 * da, c_s - 0.3 * dc, "recta mala", NARANJA),
             (a_s - 0.25 * da, c_s + 0.55 * dc, "recta regular", "#8E44AD"),
             (a_s, c_s, "recta ÓPTIMA", ROJO)]
    fig = plt.figure(figsize=(17, 10))
    xs = np.array([x.min() - 1, x.max() + 1])
    for j, (a, c, nom, col) in enumerate(casos):
        b = c - a * xm; Sk = error_cuadratico(x, y, a, b)
        ax = fig.add_subplot(2, 3, j + 1)
        ax.scatter(x, y, color=AZUL_OSC, s=35, zorder=3)
        ax.plot(xs, a * xs + b, color=col, lw=2.5)
        for xi, yi in zip(x, y):
            ax.plot([xi, xi], [yi, a * xi + b], color=col, lw=1, alpha=0.6)
        ax.set_title(f"{nom}: y = {a:.3f}x {b:+.2f}\nS = {Sk:.1f} lb²", color=col,
                     fontweight="bold", fontsize=11)
        ax.set_xlabel(ETIQUETA_X, fontsize=9); ax.set_ylabel(ETIQUETA_Y, fontsize=9)
        ax.set_ylim(y.min() - 12, y.max() + 12); ax.grid(alpha=0.4)
        ax3 = fig.add_subplot(2, 3, j + 4, projection="3d")
        ax3.computed_zorder = False
        ax3.plot_surface(A, C, S, cmap="viridis", alpha=0.35, lw=0, zorder=1)
        ax3.scatter([a_s], [c_s], [s_min], color="black", s=30, marker="x", depthshade=False, zorder=5)
        ax3.scatter([a], [c], [Sk], color=col, s=220, depthshade=False, edgecolor="white",
                    linewidth=1.5, zorder=10)
        _ejes(ax3, "a", "c = b + a·x̄", "S"); ax3.set_zlim(0, S.max())
        ax3.view_init(elev=24, azim=-55)
        ax3.set_title(f"su punto (a, b, S) en la superficie: altura {Sk:.1f}", fontsize=10)
    fig.suptitle("Cada recta es un punto de la superficie; la mejor recta es el FONDO del paraboloide",
                 fontsize=15, fontweight="bold", color=AZUL_OSC)
    fig.tight_layout()
    _guardar(fig, "15_3d_recta_y_superficie.png")


# ═════════════════ 16 · MÉTODOS NUMÉRICOS EN 3D ═════════════════
def metodos_3d(x, y, a_s, b_s, gd_simple, gd_centrado, newton):
    a_v = np.linspace(-0.3, 4.2, 220); b_v = np.linspace(-220, 15, 220)
    A, B = np.meshgrid(a_v, b_v); Z = np.log10(_S(x, y, A, B))
    fig = plt.figure(figsize=(12, 9)); ax = fig.add_subplot(111, projection="3d")
    ax.computed_zorder = False
    ax.plot_surface(A, B, Z, cmap="viridis", alpha=0.5, lw=0, rstride=2, cstride=2, zorder=1)
    for res, col, nom in [(gd_simple, NARANJA, "gradiente simple"),
                          (gd_centrado, VERDE, "gradiente con x centrada")]:
        tr = res["trayectoria"]
        pa = np.array([p[0] for p in tr]); pb = np.array([p[1] for p in tr])
        pz = np.log10(np.array([p[2] for p in tr])) + 0.05
        ax.plot(pa, pb, pz, color=col, lw=4 if col == NARANJA else 3, zorder=8,
                label=f"{nom}: {res['iteraciones']} iter. — "
                      f"{'convergió' if res['convergio'] else 'NO convergió'}")
    z0 = np.log10(error_cuadratico(x, y, 0, 0)); zf = np.log10(error_cuadratico(x, y, a_s, b_s))
    ax.plot([0, newton["a_star"]], [0, newton["b_star"]], [z0 + 0.05, zf + 0.05], color=ROJO, lw=3,
            ls="--", label="Newton: 1 paso (usa el Hessiano)")
    ax.scatter([0], [0], [z0], color="black", s=60, depthshade=False, label="inicio (0, 0)")
    ax.scatter([a_s], [b_s], [zf], color=ROJO, s=170, marker="D", depthshade=False,
               label="mínimo (a*, b*)")
    _ejes(ax, "a [lb/in]", "b [lb]", "log₁₀ S(a, b)")
    ax.view_init(elev=48, azim=-62)
    ax.set_title("Métodos numéricos descendiendo por la superficie (escala log)\n"
                 "el valle estrecho (κ ≈ 10⁶) frena al gradiente simple",
                 fontsize=13, fontweight="bold", color=AZUL_OSC)
    ax.legend(loc="upper left", fontsize=9)
    _guardar(fig, "16_3d_metodos_numericos.png")


# ═════════════════ 17 · CONTRIBUCIÓN DE CADA DATO A S ═════════════════
def contribucion_errores(x, y, a_s, b_s):
    xm, ym = x.mean(), y.mean()
    rectas = [("óptima", a_s, b_s, ROJO),
              ("pendiente 3.0", 3.0, ym - 3.0 * xm, NARANJA),
              ("óptima + 3 lb", a_s, b_s + 3, "#8E44AD")]
    fig = plt.figure(figsize=(13, 8)); ax = fig.add_subplot(111, projection="3d")
    ancho = 0.6
    for k, (nom, a, b, col) in enumerate(rectas):
        e2 = (y - a * x - b) ** 2
        ax.bar3d(x - ancho / 2, np.full_like(x, k) - 0.3, np.zeros_like(x), ancho, 0.6, e2,
                 color=col, alpha=0.8, shade=True)
    etiquetas = [f"{nom}  (S = {((y - a * x - b) ** 2).sum():.1f})" for nom, a, b, _ in rectas]
    ax.set_yticks(range(len(rectas))); ax.set_yticklabels(etiquetas, fontsize=9)
    for t, (_, _, _, col) in zip(ax.get_yticklabels(), rectas):
        t.set_color(col); t.set_fontweight("bold")
    _ejes(ax, ETIQUETA_X, "recta", "eᵢ² [lb²]")
    ax.view_init(elev=25, azim=-60)
    ax.set_title("Aporte de cada dato al error total S = Σ eᵢ²\n"
                 "la recta óptima reparte el error y logra la suma más pequeña",
                 fontsize=13, fontweight="bold", color=AZUL_OSC)
    _guardar(fig, "17_3d_contribucion_errores.png")


def generar_graficas_3d(x, y, a_s, b_s, gd_simple, gd_centrado, newton):
    s_min = error_cuadratico(x, y, a_s, b_s)
    superficie_trazas(x, y, a_s, b_s, s_min)
    curvas_nivel_3d(x, y, a_s, b_s, s_min)
    tazon_centrado(x, y, a_s, b_s, s_min)
    recta_y_superficie(x, y, a_s, b_s, s_min)
    metodos_3d(x, y, a_s, b_s, gd_simple, gd_centrado, newton)
    contribucion_errores(x, y, a_s, b_s)
    print("OK - 6 gráficas 3D guardadas en:", os.path.abspath(CARPETA))
