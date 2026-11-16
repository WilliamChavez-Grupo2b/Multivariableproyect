"""
modelo.py
---------
Define el modelo lineal  ŷ = a·x + b  y la función de error cuadrático S(a, b).

Fórmulas del documento:
  Modelo      :  ŷ_i = a·x_i + b
  Residuo     :  e_i = y_i - ŷ_i  =  y_i - a·x_i - b
  Error total :  S(a, b) = Σ (y_i - a·x_i - b)²
"""

import numpy as np


def prediccion(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Calcula los valores estimados ŷ = a·x + b para cada x_i.

    Parámetros
    ----------
    x : valores de la variable independiente
    a : pendiente de la recta
    b : intercepto (intersección con el eje vertical)

    Retorna
    -------
    Array con los valores estimados ŷ_i.
    """
    return a * x + b


def residuos(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """
    Calcula los residuos e_i = y_i - ŷ_i.

    Un residuo positivo indica que el valor real es mayor que el estimado.
    Un residuo negativo indica que el valor real es menor que el estimado.
    Un residuo cero indica que el modelo coincide exactamente con el dato.
    """
    return y - y_hat


def error_cuadratico(x: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
    """
    Calcula la función de error cuadrático total:
        S(a, b) = Σ (y_i - a·x_i - b)²

    Este es el valor que el método de mínimos cuadrados minimiza.

    Parámetros
    ----------
    x, y : datos experimentales
    a, b : parámetros actuales del modelo

    Retorna
    -------
    Valor escalar de S(a, b) ≥ 0.
    """
    e = y - a * x - b
    return float(np.sum(e ** 2))
