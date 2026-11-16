"""
graficas.py
------------
Genera las visualizaciones del proyecto:
  1. Dispersión de los datos + recta de regresión.
  2. Residuos (barras verticales desde cada punto hasta la recta).
  3. Superficie de error S(a, b) en 3D con el mínimo marcado.
  4. Curvas de nivel (contorno) de S(a, b) con la trayectoria del
     descenso por gradiente.

Las figuras se guardan en la carpeta 'salidas/' del proyecto.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from modelo import prediccion, error_cuadratico

# Carpeta de salida (relativa a la raíz del proyecto)
CARPETA_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "salidas")
os.makedirs(CARPETA_SALIDA, exist_ok=True)


def _ruta(nombre: str) -> str:
    return os.path.join(CARPETA_SALIDA, nombre)


def grafica_regresion(x, y, a, b, guardar=True) -> None:
    """
    Gráfica 1: puntos experimentales + recta ŷ = a·x + b.
    Muestra la tendencia lineal ajustada por mínimos cuadrados.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Puntos de datos
    ax.scatter(x, y, color="darkblue", s=60, zorder=3, label="Datos observados (xᵢ, yᵢ)")

    # Recta de regresión (extendida un poco más allá de los datos)
    x_min, x_max = np.min(x), np.max(x)
    margen = (x_max - x_min) * 0.08
    xs = np.linspace(x_min - margen, x_max + margen, 200)
    ys = prediccion(xs, a, b)
    ax.plot(xs, ys, color="crimson", lw=2, label=f"ŷ = {a:.4f}·x + {b:.4f}")

    ax.set_title("Regresión lineal por mínimos cuadrados")
    ax.set_xlabel("x (variable independiente)")
    ax.set_ylabel("y (variable dependiente)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if guardar:
        fig.savefig(_ruta("regresion.png"), dpi=150, bbox_inches="tight")
    plt.show()


def grafica_residuos(x, y, a, b, guardar=True) -> None:
    """
    Gráfica 2: residuos. Cada segmento vertical muestra el error e_i
    entre el punto real y la recta.
      e_i > 0 → punto por encima de la recta
      e_i < 0 → punto por debajo
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    y_hat = prediccion(x, a, b)

    # Recta
    x_min, x_max = np.min(x), np.max(x)
    xs = np.linspace(x_min, x_max, 100)
    ax.plot(xs, prediccion(xs, a, b), color="crimson", lw=2, label="Recta ajustada")

    # Segmentos de residuo
    for xi, yi, yh in zip(x, y, y_hat):
        color = "green" if yi >= yh else "orange"
        ax.plot([xi, xi], [yi, yh], color=color, lw=1.5, zorder=2)

    # Puntos
    ax.scatter(x, y, color="darkblue", s=60, zorder=3, label="Datos")

    ax.set_title(f"Residuos  eᵢ = yᵢ − ŷᵢ   (S = {error_cuadratico(x, y, a, b):.4f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if guardar:
        fig.savefig(_ruta("residuos.png"), dpi=150, bbox_inches="tight")
    plt.show()


def grafica_superficie_error(x, y, a_star=None, b_star=None, guardar=True) -> None:
    """
    Gráfica 3: superficie 3D de S(a, b) con el mínimo global marcado.
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Rango de parámetros alrededor de la solución
    a_star, b_star = resolver_rango(x, y, a_star, b_star)
    a_vals = np.linspace(a_star - 3, a_star + 3, 80)
    b_vals = np.linspace(b_star - 3, b_star + 3, 80)
    A, B = np.meshgrid(a_vals, b_vals)

    S = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            S[i, j] = error_cuadratico(x, y, A[i, j], B[i, j])

    ax.plot_surface(A, B, S, cmap="viridis", alpha=0.85, edgecolor="none")

    # Marcar el mínimo
    if a_star is not None and b_star is not None:
        s_min = error_cuadratico(x, y, a_star, b_star)
        ax.scatter([a_star], [b_star], [s_min], color="red", s=80, zorder=5,
                   label=f"Mínimo (a*={a_star:.3f}, b*={b_star:.3f})")
        ax.legend()

    ax.set_title("Superficie del error cuadrático S(a, b)")
    ax.set_xlabel("a (pendiente)")
    ax.set_ylabel("b (intercepto)")
    ax.set_zlabel("S(a, b)")

    if guardar:
        fig.savefig(_ruta("superficie_error.png"), dpi=150, bbox_inches="tight")
    plt.show()


def resolver_rango(x, y, a_star, b_star):
    """Devuelve (a_star, b_star) calculados si no se proporcionan."""
    if a_star is None or b_star is None:
        from analitico import resolver_analitico
        a_star, b_star = resolver_analitico(x, y)
    return a_star, b_star


def grafica_contorno_gradiente(x, y, a_star, b_star, historial=None, guardar=True) -> None:
    """
    Gráfica 4: curvas de nivel de S(a, b) y, opcionalmente, la trayectoria
    seguida por el descenso por gradiente hasta el mínimo.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    a_vals = np.linspace(a_star - 3, a_star + 3, 100)
    b_vals = np.linspace(b_star - 3, b_star + 3, 100)
    A, B = np.meshgrid(a_vals, b_vals)

    S = np.zeros_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            S[i, j] = error_cuadratico(x, y, A[i, j], B[i, j])

    cs = ax.contourf(A, B, S, levels=30, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="S(a, b)")
    ax.contour(A, B, S, levels=30, colors="white", linewidths=0.4, alpha=0.5)

    # Mínimo global
    ax.scatter([a_star], [b_star], color="red", s=100, marker="*", zorder=5,
               label=f"Mínimo global (a*, b*) = ({a_star:.3f}, {b_star:.3f})")

    # Trayectoria del gradiente (si se proporciona)
    if historial:
        pass  # la trayectoria completa se puede añadir registrando (a, b) por iteración

    ax.set_title("Curvas de nivel de S(a, b)")
    ax.set_xlabel("a (pendiente)")
    ax.set_ylabel("b (intercepto)")
    ax.legend()

    if guardar:
        fig.savefig(_ruta("contorno_gradiente.png"), dpi=150, bbox_inches="tight")
    plt.show()


def generar_todas(x, y, a_star, b_star, historial=None) -> None:
    """Genera todas las gráficas de una vez."""
    print("\nGenerando gráficas...")
    grafica_regresion(x, y, a_star, b_star)
    grafica_residuos(x, y, a_star, b_star)
    grafica_superficie_error(x, y, a_star, b_star)
    grafica_contorno_gradiente(x, y, a_star, b_star, historial)
    print(f"Gráficas guardadas en: {os.path.abspath(CARPETA_SALIDA)}")
