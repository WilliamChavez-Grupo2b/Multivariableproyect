"""
app_html.py
-----------
Genera results/html/app.html: la aplicación interactiva del proyecto en UN solo
archivo que funciona sin internet (RD-01).

  • Python es la fuente de verdad (RD-02): calcula los resultados oficiales
    (a*, b*, S*, Hessiano, bibliotecas, modelo cuadrático) y los incrusta como JSON.
  • El navegador recalcula con web/regresion.js (las mismas fórmulas cerradas)
    para que los controles, la edición de datos y la exclusión de puntos
    respondan al instante (RD-03).
  • Al abrir, app.html compara su cálculo con el de Python (RD-04).

Uso:
    python src/app_html.py        → escribe results/html/app.html
"""

import json
import math
import os
import sys

import numpy as np
import plotly.offline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analisis
from analitico import reporte_analitico
from datos import obtener_datos, PROBLEMA, FUENTE, NOMBRE_X, NOMBRE_Y, UNIDAD_X, UNIDAD_Y

RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
WEB = os.path.join(RAIZ, "web")
SALIDA = os.path.join(RAIZ, "results", "html", "app.html")


def _limpio(v):
    """Convierte numpy → tipos de JSON y nan/inf → None (JSON.parse no acepta NaN)."""
    if isinstance(v, dict):
        return {k: _limpio(w) for k, w in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_limpio(w) for w in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    return v


def datos_app(x: np.ndarray, y: np.ndarray) -> dict:
    """Resultados oficiales de Python que se incrustan en app.html."""
    r = reporte_analitico(x, y)
    a, b = r["a_star"], r["b_star"]
    comp = analisis.comparar_metodos(x, y)
    hv = analisis.verificar_hessiano_numerico(x, y, a, b)
    cal = analisis.calidad_ajuste(x, y, a, b)
    cq = analisis.comparar_cuadratico(x, y) if len(x) >= 4 else None
    return _limpio({
        "meta": {"problema": PROBLEMA, "fuente": FUENTE,
                 "nx": NOMBRE_X, "ny": NOMBRE_Y, "ux": UNIDAD_X, "uy": UNIDAD_Y},
        "x": x, "y": y,
        "manual": analisis.MANUAL,
        "python": {
            "a": a, "b": b, "S": r["s_min"],
            "R2": cal["R2"], "RMSE": cal["RMSE"],
            "H": hv["H"], "H_num": hv["H_num"],
            "autovalores": hv["autovalores"], "condicion": hv["condicion"],
            "bibliotecas": comp["bibliotecas"],
            "cuadratico": None if cq is None else {
                "coef": cq["cuadratico"], "coef_polyfit": cq["cuadratico_np"],
                "S": cq["S_cuad"], "F": cq["F"], "p_valor": cq["p_valor"]},
        },
    })


def construir(ruta: str = SALIDA) -> str:
    x, y = obtener_datos()
    datos = json.dumps(datos_app(x, y), ensure_ascii=False).replace("</", "<\\/")

    def leer(nombre):
        with open(os.path.join(WEB, nombre), encoding="utf-8") as f:
            return f.read()

    # Plotly va al final: así ningún marcador aparece dentro de su código
    html = (leer("plantilla.html")
            .replace("/*DATOS*/", datos)
            .replace("/*REGRESION*/", leer("regresion.js"))
            .replace("/*APP*/", leer("app.js"))
            .replace("/*PLOTLY*/", plotly.offline.get_plotlyjs()))

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(html)
    return ruta


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ruta = construir()
    print(f"[OK] {ruta}  ({os.path.getsize(ruta) / 1e6:.1f} MB)")
