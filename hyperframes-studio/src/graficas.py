"""
graficas.py — v2 (estilo científico profesional)
-------------------------------------------------
Visualizaciones del proyecto con estilo unificado:
  paneles blancos · acentos azules · escalas de color científicas.

Gráficas generadas (carpeta 'salidas/'):
  1. regresion.png            → datos + recta óptima
  2. residuos.png             → residuos eᵢ con la recta
  3. superficie_error.png     → superficie 3D ONDULADA y COLOREADA de S(a,b),
                                regiones de distinta altura y curvatura,
                                con la trayectoria del gradiente subiendo por ella
  4. contorno_gradiente.png   → curvas de nivel + trayectoria ANTES→DESPUÉS
  5. convergencia.png         → J(k) y ‖∇S‖ vs iteración (log)
  6. comparacion_antes_despues.png → coeficientes, J₀ vs J_f, ‖∇S‖, convergencia
  7. predicciones_comparacion.png  → predicciones y residuos antes vs después
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # generación headless (sin ventanas)
import matplotlib.pyplot as plt
from matplotlib import cm

from modelo import prediccion, residuos, error_cuadratico

# ── Estilo global: paneles blancos, acentos azules ──────────────────────────
AZUL      = "#0B5FCC"      # acento principal
AZUL_OSC  = "#083A80"
NARANJA   = "#E8740C"      # "antes" / contraste
ROJO      = "#D62728"      # mínimo global
GRIS      = "#5A6472"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.edgecolor":   GRIS,
    "axes.labelcolor":  AZUL_OSC,
    "axes.titlecolor":  AZUL_OSC,
    "axes.grid":        True,
    "grid.color":       "#C9D6E8",
    "grid.linewidth":   0.6,
    "grid.alpha":       0.55,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.dpi":       120,
})

# Todas las gráficas viven en results/images/ (única carpeta de resultados)
CARPETA_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "images")
os.makedirs(CARPETA_SALIDA, exist_ok=True)


def _ruta(nombre: str) -> str:
    return os.path.join(CARPETA_SALIDA, nombre)


def _superficie_S(x, y, a_vals, b_vals):
    """Matriz S[i,j] = S(a_vals[j], b_vals[i]) evaluada de forma vectorizada."""
    A, B = np.meshgrid(a_vals, b_vals)
    x = np.asarray(x, float); y = np.asarray(y, float)
    S = (((y[None, None, :] - B[..., None] - A[..., None] * x[None, None, :]) ** 2).sum(axis=-1))
    return A, B, S


def _campo_gradiente(x, y, A, B):
    """Campos ∂S/∂a y ∂S/∂b evaluados en las mallas A, B."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    dA = -2 * np.sum((y[None, None, :] - B[..., None] - A[..., None] * x[None, None, :]) * x[None, None, :], axis=-1)
    dB = -2 * np.sum((y[None, None, :] - B[..., None] - A[..., None] * x[None, None, :]), axis=-1)
    return dA, dB


# ═══════════════════════ 1 · REGRESIÓN ═══════════════════════
def grafica_regresion(x, y, a, b, guardar=True) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.asarray(x, float); y = np.asarray(y, float)
    x_min, x_max = x.min(), x.max()
    margen = (x_max - x_min) * 0.08
    xs = np.linspace(x_min - margen, x_max + margen, 200)

    # Banda de confianza visual (±1 RMSE alrededor de la recta)
    ys = prediccion(xs, a, b)
    rm = float(np.sqrt(np.mean(residuos(y, prediccion(x, a, b)) ** 2)))
    ax.fill_between(xs, ys - rm, ys + rm, color=AZUL, alpha=0.12,
                    label="Banda ±1 RMSE")
    ax.plot(xs, ys, color=AZUL, lw=2.8,
            label=f"ŷ = {a:.4f}·x + {b:.4f}   (R² máx)")
    ax.scatter(x, y, s=80, color=AZUL_OSC, edgecolor="white", lw=1.5,
               zorder=4, label="Datos observados (xᵢ, yᵢ)")

    ax.set_title("Regresión lineal por mínimos cuadrados — recta óptima")
    ax.set_xlabel("x (variable independiente)")
    ax.set_ylabel("y (variable dependiente)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    if guardar:
        fig.savefig(_ruta("regresion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════ 2 · RESIDUOS ═══════════════════════
def grafica_residuos(x, y, a, b, guardar=True) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.asarray(x, float); y = np.asarray(y, float)
    y_hat = prediccion(x, a, b)

    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, prediccion(xs, a, b), color=AZUL, lw=2.6, label="Recta ajustada")

    for xi, yi, yh in zip(x, y, y_hat):
        color = "#1B8A4C" if yi >= yh else NARANJA
        ax.plot([xi, xi], [yi, yh], color=color, lw=1.8, alpha=0.85, zorder=2)

    ax.scatter(x, y, s=80, color=AZUL_OSC, edgecolor="white", lw=1.5,
               zorder=4, label="Datos")
    ax.scatter(x, y_hat, s=45, facecolor="white", edgecolor=AZUL, lw=1.4,
               zorder=4, label="Predicciones ŷᵢ")

    ax.set_title(f"Residuos  eᵢ = yᵢ − ŷᵢ   (S = {error_cuadratico(x, y, a, b):.4f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(frameon=False)
    fig.tight_layout()
    if guardar:
        fig.savefig(_ruta("residuos.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════ 3 · SUPERFICIE 3D ONDULADA Y COLOREADA ═════════════
def grafica_superficie_error(x, y, a_star, b_star, trayectoria=None,
                             historial=None, guardar=True) -> None:
    """
    Superficie tridimensional del error S(a, b):
      · relieve ondulado con regiones de diferente altura y cambios de curvatura
      · coloración científica por escala 'viridis' según la altura S
      · el valle del mínimo marcado y la trayectoria del descenso por gradiente
        dibujada COMO CURVA 3D que desciende por la superficie (ANTES → DESPUÉS)
    """
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    a_vals = np.linspace(a_star - 2.2, a_star + 1.6, 120)
    b_vals = np.linspace(b_star - 2.2, b_star + 1.6, 120)
    A, B, S = _superficie_S(x, y, a_vals, b_vals)

    # Elevar levemente la topografía para dar relieve ondulado apreciable
    z_offset = max(0.0, float(S.min()))
    norm = matplotlib.colors.Normalize(vmin=S.min(), vmax=S.max())
    colores = cm.viridis(norm(S))

    surf = ax.plot_surface(A, B, S, facecolors=colores,
                           rstride=2, cstride=2,
                           linewidth=0.15, edgecolor="#083A8033",
                           antialiased=True, alpha=0.94)

    # Líneas de nivel proyectadas en el suelo del gráfico (relieve extra)
    nivel_base = S.min() - 0.18 * (S.max() - S.min())
    ax.contour(A, B, S, levels=14, zdir="z", offset=nivel_base,
               cmap="viridis", linewidths=0.9, alpha=0.8)

    # Mínimo global
    s_min = error_cuadratico(x, y, a_star, b_star)
    ax.scatter([a_star], [b_star], [s_min], color=ROJO, s=130, marker="*",
               depthshade=False, zorder=10,
               label=f"Mínimo global  J_f = {s_min:.4f}")

    # Trayectoria del gradiente como curva 3D sobre la superficie
    if trayectoria:
        m = len(trayectoria)
        # submuestreo determinista: primeros 30 puntos + últimos 20
        idx = list(range(min(30, m))) + list(range(m - 20, m))
        idx = sorted(set(i for i in idx if 0 <= i < m))
        pa = np.array([trayectoria[i][0] for i in idx])
        pb = np.array([trayectoria[i][1] for i in idx])
        ps = np.array([trayectoria[i][2] for i in idx]) + 0.06 * (S.max() - S.min())
        ax.plot(pa, pb, ps, color="white", lw=1.4, alpha=0.85, zorder=9)
        ax.plot(pa, pb, ps, color=AZUL, lw=2.4, zorder=9,
                label="Trayectoria del gradiente (a₀,b₀)→(a*,b*)")
        ax.scatter([pa[0]], [pb[0]], [ps[0]], color=NARANJA, s=90,
                   depthshade=False, zorder=10, label="INICIO (a₀, b₀)")

    mappable = cm.ScalarMappable(norm=norm, cmap="viridis")
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.55, pad=0.08)
    cbar.set_label("Altura  S(a, b)  — función de error", color=AZUL_OSC)

    ax.set_title("Superficie del error S(a, b) — relieve 3D coloreado\n"
                 "regiones de diferente altura y cambios de curvatura",
                 pad=14)
    ax.set_xlabel("a (pendiente)")
    ax.set_ylabel("b (intercepto)")
    ax.set_zlabel("S(a, b)")
    ax.view_init(elev=28, azim=-58)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    if guardar:
        fig.savefig(_ruta("superficie_error.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════ 4 · CONTORNO + TRAYECTORIA DEL GRADIENTE ═════════════
def grafica_contorno_gradiente(x, y, a_star, b_star, trayectoria=None,
                               guardar=True) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))

    a_vals = np.linspace(a_star - 2.2, a_star + 1.6, 200)
    b_vals = np.linspace(b_star - 2.2, b_star + 1.6, 200)
    A, B, S = _superficie_S(x, y, a_vals, b_vals)

    cs = ax.contourf(A, B, S, levels=40, cmap="viridis")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("S(a, b)", color=AZUL_OSC)
    ax.contour(A, B, S, levels=40, colors="white", linewidths=0.4, alpha=0.4)

    # Campo de gradiente (flechas hacia el mínimo)
    dA, dB = _campo_gradiente(x, y, A[::14, ::14], B[::14, ::14])
    mag = np.hypot(dA, dB); mag[mag == 0] = 1
    ax.quiver(A[::14, ::14], B[::14, ::14], -dA / mag, -dB / mag,
              color="white", alpha=0.35, scale=30, width=0.004)

    # Mínimo global
    ax.scatter([a_star], [b_star], color=ROJO, s=260, marker="*",
               zorder=6, edgecolor="white", lw=0.8,
               label=f"Mínimo global (a*, b*) = ({a_star:.3f}, {b_star:.3f})")

    # Trayectoria ANTES → DESPUÉS
    if trayectoria:
        m = len(trayectoria)
        idx = list(range(min(30, m))) + list(range(m - 20, m))
        idx = sorted(set(i for i in idx if 0 <= i < m))
        pa = [trayectoria[i][0] for i in idx]
        pb = [trayectoria[i][1] for i in idx]
        ax.plot(pa, pb, color=AZUL, lw=2.6, marker="o", ms=4,
                mec="white", mew=0.6, zorder=5,
                label=f"Descenso por gradiente ({m} iteraciones)")
        ax.scatter([pa[0]], [pb[0]], color=NARANJA, s=120, zorder=6,
                   edgecolor="white", label=f"INICIO (a₀,b₀) = ({pa[0]:.2f}, {pb[0]:.2f})")

    ax.set_title("Curvas de nivel de S(a, b) + campo de gradiente")
    ax.set_xlabel("a (pendiente)")
    ax.set_ylabel("b (intercepto)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    if guardar:
        fig.savefig(_ruta("contorno_gradiente.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════ 5 · CONVERGENCIA ═════════════
def grafica_convergencia(trayectoria, guardar=True) -> None:
    """J(k) y ‖∇S‖ contra iteración (escala log) — evidencia de convergencia."""
    k   = np.arange(len(trayectoria))
    Js  = np.array([p[2] for p in trayectoria])
    g   = np.array([max(p[3], 1e-300) for p in trayectoria])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

    ax1.plot(k, Js, color=AZUL, lw=2.4)
    ax1.set_yscale("log")
    ax1.scatter([k[0], k[-1]], [Js[0], Js[-1]], color=[NARANJA, ROJO], s=90,
                zorder=5, edgecolor="white")
    ax1.annotate(f"J₀ = {Js[0]:.2f}", (k[0], Js[0]), xytext=(14, 4),
                 textcoords="offset points", color=NARANJA, fontweight="bold")
    ax1.annotate(f"J_f = {Js[-1]:.4f}", (k[-1], Js[-1]), xytext=(-90, 16),
                 textcoords="offset points", color=ROJO, fontweight="bold")
    ax1.set_title("Convergencia de la función de error  J(k)")
    ax1.set_xlabel("Iteración k")
    ax1.set_ylabel("S(a, b) — escala log")

    ax2.plot(k, g, color=AZUL_OSC, lw=2.0)
    ax2.set_yscale("log")
    ax2.axhline(1e-8, color=ROJO, ls="--", lw=1.4,
                label="Tolerancia 1e-8")
    ax2.set_title("Norma del gradiente  ‖∇S(a, b)‖ → 0")
    ax2.set_xlabel("Iteración k")
    ax2.set_ylabel("‖∇S‖ — escala log")
    ax2.legend(frameon=False)

    fig.tight_layout()
    if guardar:
        fig.savefig(_ruta("convergencia.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ═════════════ 6 · COMPARACIÓN ANTES / DESPUÉS ═════════════
def grafica_comparacion_antes_despues(x, y, a_star, b_star, a0, b0,
                                      trayectoria, guardar=True) -> None:
    """
    Panel comparativo ANTES vs DESPUÉS:
      · coeficientes iniciales vs optimizados
      · J₀ vs J_f      · ‖∇S‖ inicial vs final      · convergencia
    """
    J0 = error_cuadratico(x, y, a0, b0)
    Jf = error_cuadratico(x, y, a_star, b_star)
    g0 = trayectoria[0][3] if len(trayectoria) > 0 else 0.0
    gf = np.hypot(*__import__("derivadas").gradiente(x, y, a_star, b_star))
    Js = np.array([p[2] for p in trayectoria])

    fig = plt.figure(figsize=(13.5, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # — Coeficientes —
    ax = fig.add_subplot(gs[0, 0])
    xpos = np.arange(2)
    v_antes  = [abs(a0), abs(b0)]
    v_desp   = [abs(a_star), abs(b_star)]
    piso = max(max(v_desp), 1e-3)
    v_antes_plot  = [max(v, piso * 0.04) for v in v_antes]   # visibles aunque sean 0
    ax.bar(xpos - 0.20, v_antes_plot, width=0.38, color=NARANJA,
           alpha=0.9, label="Antes (inicial)")
    ax.bar(xpos + 0.20, v_desp, width=0.38, color=AZUL,
           alpha=0.9, label="Después (optimizado)")
    for i, (v0, v1) in enumerate(zip(v_antes, v_desp)):
        ax.text(i - 0.20, v_antes_plot[i] * 1.04, f"{v0:.3f}", ha="center",
                color=NARANJA, fontsize=9, fontweight="bold")
        ax.text(i + 0.20, v1 * 1.04, f"{v1:.3f}", ha="center", color=AZUL,
                fontsize=9, fontweight="bold")
    ax.set_xticks(xpos); ax.set_xticklabels(["a", "b"])
    ax.set_ylim(0, piso * 1.3)
    ax.set_title("Coeficientes: iniciales vs optimizados")
    ax.legend(frameon=False, fontsize=9)

    # — J₀ vs J_f —
    ax = fig.add_subplot(gs[0, 1])
    ax.bar(["J₀\nantes", "J_f\ndespués"], [J0, Jf], color=[NARANJA, AZUL])
    reduc = 100 * (1 - Jf / J0) if J0 > 0 else 100.0
    ax.text(0, J0 * 1.03, f"{J0:.2f}", ha="center", color=NARANJA, fontweight="bold")
    ax.text(1, Jf * 1.03 + J0 * 0.01, f"{Jf:.4f}", ha="center", color=AZUL, fontweight="bold")
    ax.set_ylim(0, J0 * 1.18)
    ax.set_title(f"Función de error   ↓ {reduc:.1f}% de reducción")

    # — Gradiente —
    ax = fig.add_subplot(gs[0, 2])
    v0, v1 = max(g0, 1e-16), max(gf, 1e-16)
    ax.bar(["antes", "después"], [v0, v1], color=[NARANJA, AZUL])
    ax.set_yscale("log")
    ax.text(0, v0 * 1.4, f"{g0:.2e}", ha="center", color=NARANJA, fontsize=9)
    ax.text(1, v1 * 1.4, f"{gf:.2e}", ha="center", color=AZUL, fontsize=9)
    ax.set_ylim(v1 * 0.5, v0 * 12)
    ax.set_title("‖∇S‖: mayor → próximo a cero")

    # — Convergencia —
    ax = fig.add_subplot(gs[1, :])
    k = np.arange(len(Js))
    ax.plot(k, Js, color=AZUL, lw=2.4)
    ax.set_yscale("log")
    ax.fill_between(k, Js, Js.min() * 0.9, color=AZUL, alpha=0.10)
    ax.scatter([k[0], k[-1]], [Js[0], Js[-1]], color=[NARANJA, ROJO], s=90,
               zorder=5, edgecolor="white")
    ax.annotate(f"ANTES  J₀ = {Js[0]:.2f}", (k[0], Js[0]), xytext=(16, 4),
                textcoords="offset points", color=NARANJA, fontweight="bold")
    ax.annotate(f"DESPUÉS  J_f = {Js[-1]:.4f}  — convergencia alcanzada ✓",
                (k[-1], Js[-1]), xytext=(-330, 28),
                textcoords="offset points", color=ROJO, fontweight="bold")
    ax.set_title("Trayectoria de convergencia: de la condición inicial al mínimo global")
    ax.set_xlabel("Iteración k")
    ax.set_ylabel("S(a, b) — escala log")

    fig.suptitle("COMPARACIÓN ANTES / DESPUÉS DE LA OPTIMIZACIÓN",
                 fontsize=15, fontweight="bold", color=AZUL_OSC)
    if guardar:
        fig.savefig(_ruta("comparacion_antes_despues.png"), dpi=150,
                    bbox_inches="tight")
    plt.close(fig)


# ═════════════ 7 · PREDICCIONES Y RESIDUOS ANTES / DESPUÉS ═════════════
def grafica_predicciones_comparacion(x, y, a_star, b_star, a0, b0,
                                     guardar=True) -> None:
    """Predicciones iniciales vs ajustadas, y residuos iniciales vs finales."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    yh0 = prediccion(x, a0, b0)
    yhf = prediccion(x, a_star, b_star)
    e0  = residuos(y, yh0)
    ef  = residuos(y, yhf)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # — Predicciones —
    xs = np.linspace(x.min() - 0.3, x.max() + 0.3, 100)
    ax1.scatter(x, y, s=85, color=AZUL_OSC, edgecolor="white", lw=1.4,
                zorder=5, label="Datos reales")
    ax1.plot(xs, prediccion(xs, a0, b0), color=NARANJA, lw=2.0, ls="--",
             label=f"ANTES: ŷ = {a0:.3f}·x + {b0:.3f}")
    ax1.plot(xs, prediccion(xs, a_star, b_star), color=AZUL, lw=2.8,
             label=f"DESPUÉS: ŷ = {a_star:.3f}·x + {b_star:.3f}")
    ax1.set_title("Predicciones: iniciales vs ajustadas")
    ax1.set_xlabel("x"); ax1.set_ylabel("y")
    ax1.legend(frameon=False, fontsize=9)

    # — Residuos —
    idx = np.arange(len(x))
    ax2.bar(idx - 0.20, e0, width=0.38, color=NARANJA, alpha=0.85,
            label="Residuos iniciales")
    ax2.bar(idx + 0.20, ef, width=0.38, color=AZUL, alpha=0.9,
            label="Residuos finales")
    ax2.axhline(0, color=GRIS, lw=1)
    ax2.set_title(f"Residuos: ANTES Σe²={np.sum(e0**2):.2f}   →   DESPUÉS Σe²={np.sum(ef**2):.4f}")
    ax2.set_xlabel("Observación i"); ax2.set_ylabel("eᵢ = yᵢ − ŷᵢ")
    ax2.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    if guardar:
        fig.savefig(_ruta("predicciones_comparacion.png"), dpi=150,
                    bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════ ORQUESTADOR ═══════════════════════
def generar_todas(x, y, a_star, b_star, historial=None,
                  trayectoria=None, a_init=0.0, b_init=0.0) -> None:
    """Genera todas las gráficas (2D, 3D y comparación antes/después)."""
    print("\nGenerando gráficas (estilo científico v2)...")
    grafica_regresion(x, y, a_star, b_star)
    grafica_residuos(x, y, a_star, b_star)
    grafica_superficie_error(x, y, a_star, b_star, trayectoria, historial)
    grafica_contorno_gradiente(x, y, a_star, b_star, trayectoria)
    if trayectoria and len(trayectoria) > 1:
        grafica_convergencia(trayectoria)
        grafica_comparacion_antes_despues(x, y, a_star, b_star,
                                          a_init, b_init, trayectoria)
    grafica_predicciones_comparacion(x, y, a_star, b_star, a_init, b_init)
    print("OK - 7 graficas guardadas en:", os.path.abspath(CARPETA_SALIDA))
