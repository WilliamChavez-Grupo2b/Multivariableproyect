"""
derivadas.py
------------
Calcula las derivadas parciales de S(a,b) y la matriz Hessiana.

Fórmulas del desarrollo matemático:

  ∂S/∂a  = -2 · Σ x_i·(y_i - a·x_i - b)
  ∂S/∂b  = -2 · Σ (y_i - a·x_i - b)

  Segundas derivadas (para la Hessiana):
    ∂²S/∂a²   = 2 · Σ x_i²
    ∂²S/∂b²   = 2n
    ∂²S/∂a∂b  = 2 · Σ x_i

  Matriz Hessiana:
    H = | 2·Σx²   2·Σx |
        | 2·Σx    2n   |

  det(H) = 4·[ n·Σx² - (Σx)² ]   →  > 0 si los x_i no son todos iguales
           => mínimo global confirmado.
"""

import numpy as np


def gradiente(x: np.ndarray, y: np.ndarray, a: float, b: float) -> tuple[float, float]:
    """
    Calcula el gradiente de S(a,b) en el punto (a, b).

    Retorna
    -------
    (dS_da, dS_db) : derivadas parciales evaluadas en (a, b).
    """
    e = y - a * x - b          # residuos vectorizados
    dS_da = float(-2.0 * np.sum(x * e))
    dS_db = float(-2.0 * np.sum(e))
    return dS_da, dS_db


def hessiana(x: np.ndarray) -> np.ndarray:
    """
    Construye la matriz Hessiana de S(a,b).

    La Hessiana no depende de los valores actuales de a y b,
    sólo de los datos x_i.

    Retorna
    -------
    H : np.ndarray de forma (2, 2).
    """
    n = len(x)
    sum_x  = np.sum(x)
    sum_x2 = np.sum(x ** 2)

    H = np.array([
        [2.0 * sum_x2, 2.0 * sum_x],
        [2.0 * sum_x,  2.0 * n    ]
    ])
    return H


def verificar_minimo(x: np.ndarray) -> dict:
    """
    Verifica que el punto crítico sea un mínimo global comprobando
    que la Hessiana sea definida positiva:
      1. h11 = 2·Σx² > 0
      2. det(H) = 4·[n·Σx² - (Σx)²] > 0

    Retorna un dict con los valores y el resultado de la verificación.
    """
    n = len(x)
    H = hessiana(x)
    h11 = H[0, 0]
    det_H = np.linalg.det(H)

    # Expresión analítica del determinante para mostrar en el reporte
    det_analitico = 4.0 * (n * np.sum(x ** 2) - np.sum(x) ** 2)

    return {
        "H": H,
        "h11": h11,
        "det_H": det_analitico,
        "cond1_ok": h11 > 0,
        "cond2_ok": det_analitico > 0,
        "es_minimo": (h11 > 0) and (det_analitico > 0),
    }


def imprimir_hessiana(x: np.ndarray) -> None:
    """Imprime la Hessiana y el resultado de la verificación de mínimo."""
    info = verificar_minimo(x)
    H = info["H"]
    print("\n── Matriz Hessiana ─────────────────────────────")
    print(f"  H = | {H[0,0]:>10.4f}  {H[0,1]:>10.4f} |")
    print(f"      | {H[1,0]:>10.4f}  {H[1,1]:>10.4f} |")
    print(f"\n  Condición 1 → ∂²S/∂a² = {info['h11']:.4f} > 0 : {info['cond1_ok']}")
    print(f"  Condición 2 → det(H)  = {info['det_H']:.4f} > 0 : {info['cond2_ok']}")
    if info["es_minimo"]:
        print("  ✓ La Hessiana es definida positiva → mínimo global confirmado.")
    else:
        print("  ✗ No se puede confirmar el mínimo (¿todos los x_i son iguales?).")
    print("─" * 50)
