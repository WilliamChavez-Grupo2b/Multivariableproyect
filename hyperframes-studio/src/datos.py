"""
datos.py
--------
Carga y valida el conjunto de datos REALES del problema (ver data/input/FUENTE.md):

    <raíz>/data/input/datos.csv  (raíz = hyperframes-studio)

Formato esperado (columnas x,y):
    x,y
    1.000000,3.123456
    ...

Validaciones aplicadas (carga y validación de datos):
  • el archivo existe
  • encabezados correctos (columnas "x" e "y")
  • todos los valores son numéricos y finitos
  • al menos 2 observaciones
  • no todos los x son iguales (si no, la pendiente no es única)
"""

import os
import csv
import numpy as np

RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUTA_CSV = os.path.join(RAIZ, "data", "input", "datos.csv")

# ── Descripción del problema real (ver data/input/FUENTE.md) ────────────────
PROBLEMA = "Estatura vs. peso promedio de mujeres estadounidenses (30–39 años)"
NOMBRE_X = "Estatura"
NOMBRE_Y = "Peso promedio"
UNIDAD_X = "in"
UNIDAD_Y = "lb"
ETIQUETA_X = f"{NOMBRE_X} x [{UNIDAD_X}]"
ETIQUETA_Y = f"{NOMBRE_Y} y [{UNIDAD_Y}]"
FUENTE = "The World Almanac and Book of Facts (1975) — dataset 'women' de R"


def cargar_csv(ruta=RUTA_CSV) -> tuple[np.ndarray, np.ndarray]:
    """Lee el CSV y devuelve los arrays (X, Y). Lanza errores descriptivos."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el archivo de datos: {ruta}\n"
            "Revisa data/input/FUENTE.md para restaurar los datos reales."
        )

    xs, ys = [], []
    with open(ruta, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        encabezados = [c.strip().lower() for c in next(reader)]

        if "x" not in encabezados or "y" not in encabezados:
            raise ValueError(f"El CSV debe tener las columnas 'x' e 'y'. Encontrado: {encabezados}")

        ix = encabezados.index("x")
        iy = encabezados.index("y")

        for num_fila, fila in enumerate(reader, start=2):
            if not fila or all(not c.strip() for c in fila):
                continue    # saltar filas vacías
            try:
                x = float(fila[ix])
                y = float(fila[iy])
            except (ValueError, IndexError):
                raise ValueError(f"Fila {num_fila}: valor no numérico → {fila}")
            if not (np.isfinite(x) and np.isfinite(y)):
                raise ValueError(f"Fila {num_fila}: valor no finito → {fila}")
            xs.append(x)
            ys.append(y)

    return np.array(xs), np.array(ys)


def validar_datos(x: np.ndarray, y: np.ndarray) -> None:
    """Valida las condiciones matemáticas del problema."""
    if len(x) != len(y):
        raise ValueError("X e Y deben tener la misma cantidad de elementos.")
    if len(x) < 2:
        raise ValueError("Se necesitan al menos 2 observaciones.")
    if np.all(x == x[0]):
        raise ValueError(
            "Todos los valores de x son iguales: la pendiente y el intercepto "
            "no pueden determinarse de manera única con estos datos."
        )


def obtener_datos() -> tuple[np.ndarray, np.ndarray]:
    """Carga + valida. Es el punto de entrada usado por todo el proyecto."""
    x, y = cargar_csv()
    validar_datos(x, y)
    return x, y


def resumen(x: np.ndarray, y: np.ndarray) -> None:
    """Imprime un resumen tabular de los datos cargados."""
    n = len(x)
    print(f"\n{'─'*35}")
    print(f"  {PROBLEMA}")
    print(f"  Fuente: {FUENTE}")
    print(f"  DATOS (desde CSV)  (n = {n} observaciones)")
    print(f"{'─'*35}")
    print(f"  {'i':>3}  {'x_i ['+UNIDAD_X+']':>10}  {'y_i ['+UNIDAD_Y+']':>10}")
    print(f"{'─'*35}")
    for i, (xi, yi) in enumerate(zip(x, y), start=1):
        print(f"  {i:>3}  {xi:>10.4f}  {yi:>10.4f}")
    print(f"{'─'*35}\n")
