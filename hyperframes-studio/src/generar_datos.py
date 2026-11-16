"""
generar_datos.py
----------------
Genera automáticamente el conjunto de datos del problema y lo guarda en:

    <raíz>/hyperframes-studio/data/input/datos.csv

El problema del modelo lineal  y = a·x + b  tiene TRES variables:

  1. x : variable independiente (entrada)
  2. y : variable dependiente (respuesta observada)
  3. e : error/residuo  e_i = y_i − (a·x_i + b)   → componente aleatoria

Los datos se generan con ruido gaussiano DETERMINISTA (semilla fija),
de modo que el proyecto sea reproducible: mismos datos en cada ejecución.

Ejecución:
    python generar_datos.py            → usa los parámetros por defecto
    python generar_datos.py 8 2.5 1.0  # n=8, a_real=2.5, b_real=1.0
"""

import os
import sys
import csv
import numpy as np

# Consolas de Windows (cp1252): forzar UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Rutas
# Rutas: este script vive en hyperframes-studio/src → la raíz es el padre
RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CARPETA_SALIDA = os.path.join(RAIZ, "data", "input")
ARCHIVO_CSV = os.path.join(CARPETA_SALIDA, "datos.csv")

# Parámetros del generador (pueden sobrescribirse por línea de comandos)
# SEMILLA = None → NUEVOS datos en cada ejecución (aleatorio por reloj del sistema).
# Passa una semilla fija por CLI (ej: python generar_datos.py 10 2 1.5 0.8 42) para reproducibilidad.
SEMILLA = None
N_PUNTOS = 10           # cantidad de observaciones
A_REAL = 2.0            # pendiente "real" del fenómeno
B_REAL = 1.5            # intercepto "real" del fenómeno
RUIDO_DESV = 0.8        # desviación estándar del ruido gaussiano (e)


def generar_datos(n=N_PUNTOS, a_real=A_REAL, b_real=B_REAL,
                  ruido=RUIDO_DESV, semilla=SEMILLA):
    """
    Genera n observaciones:
        x_i   : valores espaciados uniformemente de 1 a n
        y_i   : a_real·x_i + b_real + ruido gaussiano (semilla fija)
    """
    rng = np.random.default_rng(semilla)          # None → semilla del sistema (datos nuevos)
    x = np.linspace(1.0, float(n), n)
    e = rng.normal(0.0, ruido, size=n)            # error aleatorio (e)
    y = a_real * x + b_real + e
    return x, y, e


def guardar_csv(x, y, ruta=ARCHIVO_CSV):
    """Escribe el CSV con encabezados x,y usado por todo el pipeline."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for xi, yi in zip(x, y):
            writer.writerow([f"{xi:.6f}", f"{yi:.6f}"])
    return ruta


def imprimir_definicion_problema(n, a_real, b_real, ruta, semilla=None):
    """Imprime la definición del problema y sus tres variables."""
    print("═" * 58)
    print("  DEFINICIÓN DEL PROBLEMA")
    print("═" * 58)
    print("""
  Modelo propuesto:   y = a·x + b

  Tres variables del modelo:
    1)  x  → variable independiente (entrada del modelo)
    2)  y  → variable dependiente  (respuesta observada)
    3)  e  → error o residuo:  e_i = y_i − (a·x_i + b)

  Objetivo: encontrar a* y b* que minimicen
        S(a,b) = Σ (y_i − a·x_i − b)²
""")
    print(f"  Parámetros de generación: n={n}, a_real={a_real}, "
          f"b_real={b_real}, semilla={semilla if semilla is not None else 'aleatoria (datos NUEVOS)'}")
    print(f"  CSV generado en: {ruta}")
    print("═" * 58)


def main(argv):
    # Parámetros opcionales: n, a_real, b_real, ruido, semilla
    n       = int(argv[1])   if len(argv) > 1 else N_PUNTOS
    a_real  = float(argv[2]) if len(argv) > 2 else A_REAL
    b_real  = float(argv[3]) if len(argv) > 3 else B_REAL
    ruido   = float(argv[4]) if len(argv) > 4 else RUIDO_DESV
    semilla = int(argv[5])   if len(argv) > 5 else SEMILLA   # None → datos nuevos

    x, y, e = generar_datos(n, a_real, b_real, ruido, semilla)
    ruta = guardar_csv(x, y)
    imprimir_definicion_problema(n, a_real, b_real, ruta, semilla)
    return ruta


if __name__ == "__main__":
    main(sys.argv)
