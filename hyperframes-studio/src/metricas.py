"""
metricas.py
------------
Calcula métricas de calidad del ajuste lineal y analiza la relación
entre las variables (positiva, negativa o débil).

Métricas implementadas:

  R²  (coeficiente de determinación):
      R² = 1 − SS_res / SS_tot
      SS_res = Σ (y_i − ŷ_i)²          (suma de cuadrados de residuos)
      SS_tot = Σ (y_i − ȳ)²            (suma de cuadrados total)

  MSE  (error cuadrático medio)   : SS_res / n
  RMSE (raíz del error cuadrático medio): √MSE
  MAE  (error absoluto medio)     : Σ |e_i| / n

Análisis de la pendiente (según las conclusiones del documento):
  a* > 0        → relación positiva: cuando x aumenta, y también aumenta.
  a* < 0        → relación negativa (inversa): cuando x aumenta, y disminuye.
  a* ≈ 0        → relación lineal débil o inexistente.
"""

import warnings

import numpy as np

from modelo import prediccion, residuos


def r_cuadrado(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Coeficiente de determinación R² ∈ [0, 1]. Cerca de 1 = buen ajuste.

    Si todos los y son iguales (SS_tot = 0), R² no está definido: devuelve nan.
    """
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        warnings.warn("R² no está definido: todos los valores de y son iguales (SS_tot = 0).")
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Error cuadrático medio: promedio de los errores al cuadrado."""
    e = residuos(y, y_hat)
    return float(np.mean(e ** 2))


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Raíz del error cuadrático medio (en las unidades de y)."""
    return float(np.sqrt(mse(y, y_hat)))


def mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Error absoluto medio."""
    e = residuos(y, y_hat)
    return float(np.mean(np.abs(e)))


def analizar_pendiente(a: float, umbral: float = 1e-6) -> str:
    """
    Clasifica la relación entre las variables según el signo de la pendiente.

    Retorna una descripción textual del tipo de relación.
    """
    if a > umbral:
        return (
            "Relación POSITIVA (a* > 0): cuando x aumenta, y tiende a aumentar. "
            "La recta sube de izquierda a derecha."
        )
    elif a < -umbral:
        return (
            "Relación NEGATIVA (a* < 0): cuando x aumenta, y tiende a disminuir. "
            "La recta baja de izquierda a derecha (relación inversa)."
        )
    else:
        return (
            "Relación lineal DÉBIL o INEXISTENTE (a* ≈ 0): la recta es "
            "aproximadamente horizontal; no hay tendencia lineal clara."
        )


def tabla_residuos(x: np.ndarray, y: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Construye una tabla con: x_i, y_i, ŷ_i, e_i = y_i − ŷ_i.
    El signo de e_i indica si el punto está por encima (e>0),
    por debajo (e<0) o sobre la recta (e=0).
    """
    y_hat = prediccion(x, a, b)
    e = residuos(y, y_hat)
    return np.column_stack([x, y, y_hat, e])


def imprimir_metricas(x: np.ndarray, y: np.ndarray, a: float, b: float) -> dict:
    """Imprime todas las métricas y la tabla de residuos. Retorna el diccionario."""
    y_hat = prediccion(x, a, b)
    e = residuos(y, y_hat)

    resultados = {
        "R2":  r_cuadrado(y, y_hat),
        "MSE": mse(y, y_hat),
        "RMSE": rmse(y, y_hat),
        "MAE": mae(y, y_hat),
        "tabla": tabla_residuos(x, y, a, b),
        "relacion": analizar_pendiente(a),
        "e_max_pos": float(np.max(e)),    # punto más alejado por encima
        "e_max_neg": float(np.min(e)),    # punto más alejado por debajo
    }

    print("\n" + "═" * 55)
    print("  MÉTRICAS DE AJUSTE")
    print("═" * 55)
    print(f"\n  R²   = {resultados['R2']:.6f}")
    print(f"  MSE  = {resultados['MSE']:.6f}")
    print(f"  RMSE = {resultados['RMSE']:.6f}")
    print(f"  MAE  = {resultados['MAE']:.6f}")

    print("\n" + "─" * 55)
    print("  TABLA DE RESIDUOS")
    print("─" * 55)
    print(f"  {'x_i':>8}  {'y_i':>8}  {'ŷ_i':>8}  {'e_i':>8}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for xi, yi, yh, ei in resultados["tabla"]:
        print(f"  {xi:>8.4f}  {yi:>8.4f}  {yh:>8.4f}  {ei:>8.4f}")
    print("─" * 55)

    print("\n• Interpretación de los residuos:")
    print(f"  Máximo residuo positivo (+)  : {resultados['e_max_pos']:.4f}  → punto por ENCIMA de la recta")
    print(f"  Máximo residuo negativo (−)  : {resultados['e_max_neg']:.4f}  → punto por DEBAJO de la recta")

    print("\n• Interpretación de la pendiente:")
    print(f"  {resultados['relacion']}")
    print("═" * 55)

    return resultados
