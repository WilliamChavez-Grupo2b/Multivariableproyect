"""
gradiente.py
------------
Implementa descenso por gradiente para minimizar S(a, b) numéricamente,
llegando al mismo resultado que la solución analítica y demostrando que
el punto crítico es efectivamente el mínimo global.

El gradiente de S(a, b) es:
  ∂S/∂a = -2 · Σ x_i · (y_i - a·x_i - b)
  ∂S/∂b = -2 · Σ (y_i - a·x_i - b)

Regla de actualización:
  a ← a − η · ∂S/∂a
  b ← b − η · ∂S/∂b

donde η (eta) es la tasa de aprendizaje.
"""

import numpy as np

from modelo import error_cuadratico
from derivadas import gradiente as calc_gradiente


def descenso_gradiente(
    x: np.ndarray,
    y: np.ndarray,
    a_init: float = 0.0,
    b_init: float = 0.0,
    eta: float = 0.01,
    max_iter: int = 10_000,
    tolerancia: float = 1e-8,
) -> dict:
    """
    Minimiza S(a, b) mediante descenso por gradiente.

    Parámetros
    ----------
    x, y        : datos experimentales
    a_init      : valor inicial de a
    b_init      : valor inicial de b
    eta         : tasa de aprendizaje (paso)
    max_iter    : número máximo de iteraciones
    tolerancia  : criterio de parada (norma del gradiente < tolerancia)

    Retorna
    -------
    dict con:
      a_star     : pendiente encontrada
      b_star     : intercepto encontrado
      iteraciones: número de iteraciones realizadas
      historial  : lista de (iteración, S, norma_grad) cada 500 pasos
      convergio  : bool
    """
    a, b = float(a_init), float(b_init)
    historial = []
    dS_da0, dS_db0 = calc_gradiente(x, y, a, b)
    trayectoria = [(a, b, error_cuadratico(x, y, a, b),
                    (dS_da0 ** 2 + dS_db0 ** 2) ** 0.5)]

    for it in range(1, max_iter + 1):
        dS_da, dS_db = calc_gradiente(x, y, a, b)

        norma = (dS_da ** 2 + dS_db ** 2) ** 0.5

        # Registrar cada 500 iteraciones o en la primera
        if it == 1 or it % 500 == 0:
            S = error_cuadratico(x, y, a, b)
            historial.append((it, S, norma))

        # Criterio de convergencia
        if norma < tolerancia:
            S = error_cuadratico(x, y, a, b)
            historial.append((it, S, norma))
            return {
                "a_star": a,
                "b_star": b,
                "iteraciones": it,
                "historial": historial,
                "trayectoria": trayectoria,
                "convergio": True,
            }

        # Actualización
        a -= eta * dS_da
        b -= eta * dS_db
        trayectoria.append((a, b, error_cuadratico(x, y, a, b), norma))

    return {
        "a_star": a,
        "b_star": b,
        "iteraciones": max_iter,
        "historial": historial,
        "trayectoria": trayectoria,
        "convergio": False,
    }


def ajustar_eta(x: np.ndarray) -> float:
    """
    Sugiere una tasa de aprendizaje segura basada en la escala de los datos.
    Usa η = 1 / (2·Σx²) como cota superior de convergencia garantizada.
    """
    sum_x2 = float(np.sum(x ** 2))
    n = len(x)
    return 1.0 / (4.0 * (sum_x2 + n))


def imprimir_resultado_gradiente(res: dict) -> None:
    """Imprime el resultado del descenso por gradiente."""
    print("\n" + "═" * 55)
    print("  DESCENSO POR GRADIENTE")
    print("═" * 55)

    print("\n• Historial de convergencia:")
    print(f"  {'Iter':>7}  {'S(a,b)':>14}  {'‖∇S‖':>14}")
    print(f"  {'─'*7}  {'─'*14}  {'─'*14}")
    for (it, S, norma) in res["historial"][-8:]:    # últimas 8 entradas
        print(f"  {it:>7}  {S:>14.6f}  {norma:>14.8f}")

    print(f"\n• Iteraciones totales : {res['iteraciones']}")
    print(f"• Convergió           : {res['convergio']}")
    print(f"\n• Parámetros encontrados:")
    print(f"  a* = {res['a_star']:.6f}")
    print(f"  b* = {res['b_star']:.6f}")
    print("═" * 55)
