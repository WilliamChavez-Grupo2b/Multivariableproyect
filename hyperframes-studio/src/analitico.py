"""
analitico.py
------------
Resuelve el sistema de ecuaciones normales de forma analítica (cerrada).

Fórmulas del desarrollo matemático:

  Ecuación normal 1 :  a·Σx²  +  b·Σx   = Σxy
  Ecuación normal 2 :  a·Σx   +  n·b    = Σy

  Pendiente óptima:
      a* = [ n·Σxy − (Σx)(Σy) ] / [ n·Σx² − (Σx)² ]

  Intercepto óptimo:
      b* = (Σy − a*·Σx) / n   =  ȳ − a*·x̄

  Recta de regresión final:
      ŷ = a*·x + b*

Estos valores corresponden al mínimo global de S(a,b) siempre que
existan al menos dos valores diferentes de x.
"""

import numpy as np

from datos import obtener_datos
from derivadas import imprimir_hessiana
from modelo import error_cuadratico


def resolver_analitico(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Calcula los parámetros óptimos (a*, b*) mediante las fórmulas cerradas.

    Parámetros
    ----------
    x, y : datos experimentales

    Retorna
    -------
    (a_star, b_star) : pendiente e intercepto óptimos.
    """
    n = len(x)

    sum_x  = np.sum(x)
    sum_y  = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    # Denominador común: n·Σx² − (Σx)²
    denominador = n * sum_x2 - sum_x ** 2

    if denominador == 0:
        raise ValueError(
            "El denominador es 0: todos los valores de x son iguales. "
            "No se puede determinar una recta única con estos datos."
        )

    a_star = (n * sum_xy - sum_x * sum_y) / denominador

    # b* = (Σy − a*·Σx) / n  =  ȳ − a*·x̄
    b_star = (sum_y - a_star * sum_x) / n

    return a_star, b_star


def resolver_sistema_matricial(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    Resuelve el sistema de ecuaciones normales usando álgebra lineal
    (equivalente a la solución cerrada, útil como verificación):

        | Σx²   Σx | | a |   | Σxy |
        | Σx    n  | | b | = | Σy  |
    """
    n = len(x)
    A = np.array([
        [np.sum(x ** 2), np.sum(x)],
        [np.sum(x),      n      ]
    ])
    B = np.array([np.sum(x * y), np.sum(y)])
    a_star, b_star = np.linalg.solve(A, B)
    return a_star, b_star


def reporte_analitico(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Genera un reporte completo de la solución analítica.
    Devuelve un diccionario con todos los resultados relevantes.
    """
    n = len(x)

    # Sumatorias
    sum_x  = np.sum(x)
    sum_y  = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)

    # Medias
    x_mean = sum_x / n
    y_mean = sum_y / n

    # Solución óptima
    a_star, b_star = resolver_analitico(x, y)

    # Verificación por sistema matricial
    a_mat, b_mat = resolver_sistema_matricial(x, y)

    # Error cuadrático mínimo
    s_min = error_cuadratico(x, y, a_star, b_star)

    return {
        "n": n,
        "sum_x": sum_x,
        "sum_y": sum_y,
        "sum_xy": sum_xy,
        "sum_x2": sum_x2,
        "x_mean": x_mean,
        "y_mean": y_mean,
        "a_star": a_star,
        "b_star": b_star,
        "a_mat": a_mat,
        "b_mat": b_mat,
        "s_min": s_min,
        "denominador": n * sum_x2 - sum_x ** 2,
    }


def imprimir_reporte(r: dict) -> None:
    """Imprime el reporte completo de la solución analítica."""
    print("\n" + "═" * 55)
    print("  SOLUCIÓN ANALÍTICA (MÍNIMOS CUADRADOS)")
    print("═" * 55)

    print("\n• Sumatorias:")
    print(f"  n       = {r['n']}")
    print(f"  Σx      = {r['sum_x']:.4f}")
    print(f"  Σy      = {r['sum_y']:.4f}")
    print(f"  Σx·y    = {r['sum_xy']:.4f}")
    print(f"  Σx²     = {r['sum_x2']:.4f}")

    print("\n• Medias:")
    print(f"  x̄ = {r['x_mean']:.4f}")
    print(f"  ȳ = {r['y_mean']:.4f}")

    print("\n• Sistema de ecuaciones normales:")
    print(f"  {r['sum_x2']:.4f}·a + {r['sum_x']:.4f}·b = {r['sum_xy']:.4f}")
    print(f"  {r['sum_x']:.4f}·a + {r['n']}·b = {r['sum_y']:.4f}")

    print("\n• Parámetros óptimos:")
    print(f"  a* (fórmula cerrada)  = {r['a_star']:.6f}")
    print(f"  b* (fórmula cerrada)  = {r['b_star']:.6f}")
    print(f"  a* (sistema lineal)   = {r['a_mat']:.6f}   [verificación]")
    print(f"  b* (sistema lineal)   = {r['b_mat']:.6f}   [verificación]")

    print("\n• Recta de regresión final:")
    print(f"  ŷ = {r['a_star']:.6f}·x + ({r['b_star']:.6f})")

    print(f"\n• Error cuadrático mínimo  S(a*, b*) = {r['s_min']:.6f}")
    print("═" * 55)


def main_analitico() -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Flujo completo de la solución analítica:
    datos → reporte → verificación Hessiana → retorna resultados.
    """
    x, y = obtener_datos()
    r = reporte_analitico(x, y)
    imprimir_reporte(r)

    print("\n• Verificación mediante la matriz Hessiana:")
    imprimir_hessiana(x)

    return x, y, r


if __name__ == "__main__":
    main_analitico()
