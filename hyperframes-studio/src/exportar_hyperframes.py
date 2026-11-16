"""
exportar_hyperframes.py
-----------------------
Exporta los resultados del análisis al proyecto HyperFrames para que el
video consuma los datos reales del pipeline Python.

Genera dos archivos:
  hyperframes-studio/data/generated/datos.json   → datos crudos + resultados
  hyperframes-studio/data/generated/datos.js     → window.HYPEFRAMES_DATA = {...}

Estructura del JSON/JS (compatible con scripts/procesar-datos.mjs):
  {
    "total": n,
    "pendiente": a*,
    "intercepto": b*,
    "errorCuadratico": S(a*,b*),
    "ecuacion": "y = a·x + b",
    "r2": ..., "mse": ..., "rmse": ..., "mae": ...,
    "relacion": "...",
    "hessiana": [[h11,h12],[h21,h22]], "detHessiana": ...,
    "gradiente": {"a": ..., "b": ..., "iteraciones": ..., "convergio": true},
    "datos": [{"x":..., "y":...}, ...]
  }
"""

import os
import json
import numpy as np

from datos import obtener_datos
from analitico import reporte_analitico
from derivadas import verificar_minimo
from gradiente import descenso_gradiente, ajustar_eta
from metricas import r_cuadrado, mse, rmse, mae, analizar_pendiente
from modelo import prediccion

RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CARPETA_SALIDA = os.path.join(RAIZ, "data", "generated")


def exportar() -> dict:
    """Ejecuta el pipeline completo y escribe los archivos de salida."""
    x, y = obtener_datos()

    r = reporte_analitico(x, y)
    h = verificar_minimo(x)

    eta = ajustar_eta(x)
    g = descenso_gradiente(x, y, eta=eta)

    a, b = r["a_star"], r["b_star"]
    y_hat = prediccion(x, a, b)

    resultado = {
        "total": int(r["n"]),
        "pendiente": a,
        "intercepto": b,
        "errorCuadratico": r["s_min"],
        "ecuacion": f"y = {a:.3f}x + {b:.3f}",
        "r2": r_cuadrado(y, y_hat),
        "mse": mse(y, y_hat),
        "rmse": rmse(y, y_hat),
        "mae": mae(y, y_hat),
        "relacion": analizar_pendiente(a),
        "hessiana": [[float(h["H"][0, 0]), float(h["H"][0, 1])],
                     [float(h["H"][1, 0]), float(h["H"][1, 1])]],
        "detHessiana": float(h["det_H"]),
        "esMinimoGlobal": bool(h["es_minimo"]),
        "gradiente": {
            "a": g["a_star"],
            "b": g["b_star"],
            "iteraciones": g["iteraciones"],
            "convergio": g["convergio"],
        },
        "datos": [{"x": float(xi), "y": float(yi)} for xi, yi in zip(x, y)],
    }

    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # JSON (datos crudos)
    with open(os.path.join(CARPETA_SALIDA, "datos.json"), "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    # JS (consumido directamente por las composiciones HyperFrames)
    with open(os.path.join(CARPETA_SALIDA, "datos.js"), "w", encoding="utf-8") as f:
        f.write(f"window.HYPEFRAMES_DATA = {json.dumps(resultado, indent=2, ensure_ascii=False)};\n")

    return resultado


def imprimir_resumen_export(res: dict) -> None:
    print("\n" + "═" * 55)
    print("  EXPORTACIÓN A HYPERFRAMES")
    print("═" * 55)
    print(f"  total           = {res['total']}")
    print(f"  pendiente a*    = {res['pendiente']:.6f}")
    print(f"  intercepto b*   = {res['intercepto']:.6f}")
    print(f"  S(a*, b*)       = {res['errorCuadratico']:.6f}")
    print(f"  ecuación        = {res['ecuacion']}")
    print(f"  R²              = {res['r2']:.6f}")
    print(f"  mín. global     = {res['esMinimoGlobal']}")
    print(f"\n  Archivos generados:")
    print(f"   → {os.path.join(CARPETA_SALIDA, 'datos.json')}")
    print(f"   → {os.path.join(CARPETA_SALIDA, 'datos.js')}")
    print("═" * 55)


if __name__ == "__main__":
    imprimir_resumen_export(exportar())
