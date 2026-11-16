"""
tests/pruebas.py
----------------
Pruebas automáticas del pipeline completo (sin dependencias externas,
usa asserts y se ejecuta con:  python pruebas.py  desde la carpeta tests).

Cubre:
  1. Generador de datos (CSV reproducible)
  2. Carga y validación de datos
  3. Función de error cuadrático
  4. Derivadas parciales (verificadas contra diferencias finitas)
  5. Ecuaciones normales (fórmula cerrada vs sistema lineal)
  6. Matriz Hessiana y mínimo global
  7. Descenso por gradiente (converge al mismo resultado)
  8. Comparación de métodos
  9. Métricas
 10. Exportación a HyperFrames (archivos generados)
"""

import os
import sys
import json
import numpy as np

# Consolas de Windows (cp1252): forzar UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, SRC)

# La raíz del proyecto ahora ES hyperframes-studio (padre de tests/)
RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from generar_datos import generar_datos, guardar_csv
from datos import cargar_csv, validar_datos, obtener_datos
from modelo import prediccion, residuos, error_cuadratico
from derivadas import gradiente as calc_gradiente, hessiana, verificar_minimo
from analitico import resolver_analitico, resolver_sistema_matricial, reporte_analitico
from gradiente import descenso_gradiente, ajustar_eta
from metricas import r_cuadrado, mse, rmse, mae, analizar_pendiente
from exportar_hyperframes import exportar

PASS = "✓"
FAIL = "✗"
resultados = []


def prueba(nombre, fn):
    try:
        fn()
        resultados.append((nombre, True, ""))
        print(f"  {PASS} {nombre}")
    except AssertionError as ex:
        resultados.append((nombre, False, str(ex)))
        print(f"  {FAIL} {nombre}  →  {ex}")
    except Exception as ex:
        resultados.append((nombre, False, f"{type(ex).__name__}: {ex}"))
        print(f"  {FAIL} {nombre}  →  {type(ex).__name__}: {ex}")


# ── 1. Generador de datos ────────────────────────────────────────────────────
def t_generador():
    # Con la MISMA semilla → reproducible; con semillas distintas → datos NUEVOS
    x1, y1, _ = generar_datos(semilla=7)
    x2, y2, _ = generar_datos(semilla=7)
    assert np.allclose(x1, x2) and np.allclose(y1, y2), "Con la misma semilla deben ser reproducibles"
    x3, y3, _ = generar_datos(semilla=8)
    assert not np.allclose(y1, y3), "Semillas distintas deben dar datos distintos"
    assert len(x1) >= 2, "Debe generar al menos 2 puntos"
    ruta = guardar_csv(x1, y1)
    assert os.path.exists(ruta), "El CSV no fue creado"

def t_generador_parametros():
    x, y, e = generar_datos(n=20, a_real=3.0, b_real=-1.0, ruido=0.0, semilla=7)
    # Sin ruido, y debe ser exactamente 3x - 1
    assert np.allclose(y, 3.0 * x - 1.0), "Sin ruido el modelo no se cumple"
    assert abs(e).sum() < 1e-12

# ── 2. Carga y validación ───────────────────────────────────────────────────
def t_carga_valida():
    x, y = obtener_datos()
    assert len(x) == len(y) and len(x) >= 2
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))

def t_validacion_detecta_x_iguales():
    try:
        validar_datos(np.array([1.0, 1.0]), np.array([1.0, 2.0, 3.0]))
        assert False, "Debía rechazar x constantes"
    except ValueError:
        pass

# ── 3. Función de error ──────────────────────────────────────────────────────
def t_error_cero_en_datos_perfectos():
    x = np.array([1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    assert abs(error_cuadratico(x, y, 2.0, 1.0)) < 1e-12, "S debe ser 0"

def t_error_propiedades():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    s_opt = error_cuadratico(x, y, 2.0, 0.0)
    s_peor = error_cuadratico(x, y, 5.0, 5.0)
    assert s_opt < s_peor, "El óptimo debe tener menor error que cualquier otro punto"

# ── 4. Derivadas parciales (vs diferencias finitas) ─────────────────────────
def t_gradiente_diferencias_finitas():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
    a, b = 1.5, 0.7
    ga, gb = calc_gradiente(x, y, a, b)
    h = 1e-7
    ga_num = (error_cuadratico(x, y, a + h, b) - error_cuadratico(x, y, a - h, b)) / (2 * h)
    gb_num = (error_cuadratico(x, y, a, b + h) - error_cuadratico(x, y, a, b - h)) / (2 * h)
    assert abs(ga - ga_num) < 1e-4, f"∂S/∂a difiere: {ga} vs {ga_num}"
    assert abs(gb - gb_num) < 1e-4, f"∂S/∂b difiere: {gb} vs {gb_num}"

def t_gradiente_cero_en_optimo():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    a_star, b_star = resolver_analitico(x, y)
    ga, gb = calc_gradiente(x, y, a_star, b_star)
    assert abs(ga) < 1e-9 and abs(gb) < 1e-9, "∇S debe ser 0 en el punto crítico"

# ── 5. Ecuaciones normales ───────────────────────────────────────────────────
def t_ecuaciones_normales_consistentes():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 15)
    y = 2.5 * x - 3.0 + rng.normal(0, 0.5, 15)
    a1, b1 = resolver_analitico(x, y)
    a2, b2 = resolver_sistema_matricial(x, y)
    assert abs(a1 - a2) < 1e-9 and abs(b1 - b2) < 1e-9

def t_recuperacion_parametros_reales():
    # Con muchos datos y poco ruido, la regresión debe recuperar a≈2, b≈1.5
    x, y, _ = generar_datos(n=500, a_real=2.0, b_real=1.5, ruido=0.05, semilla=1)
    a, b = resolver_analitico(x, y)
    assert abs(a - 2.0) < 0.05, f"pendiente recuperada {a}"
    assert abs(b - 1.5) < 0.05, f"intercepto recuperado {b}"

# ── 6. Hessiana ──────────────────────────────────────────────────────────────
def t_hessiana_simetrica_positiva():
    x = np.array([1.0, 3.0, 4.0, 7.0])
    H = hessiana(x)
    assert np.allclose(H, H.T), "H debe ser simétrica"
    info = verificar_minimo(x)
    assert info["es_minimo"], "H debe ser definida positiva"
    # det analítico: 4·(n·Σx² − (Σx)²)
    n = len(x)
    det_manual = 4 * (n * np.sum(x**2) - np.sum(x)**2)
    assert abs(info["det_H"] - det_manual) < 1e-9

def t_hessiana_coincide_con_numpy():
    x = np.array([1.0, 2.0, 3.0])
    H = hessiana(x)
    eigs = np.linalg.eigvalsh(H)
    assert np.all(eigs > 0), "Todos los autovalores deben ser positivos"

# ── 7. Descenso por gradiente ────────────────────────────────────────────────
def t_gradiente_converge():
    x, y = obtener_datos()
    eta = ajustar_eta(x)
    res = descenso_gradiente(x, y, eta=eta, max_iter=50_000)
    assert res["convergio"], "El descenso no convergió"
    a_an, b_an = resolver_analitico(x, y)
    assert abs(res["a_star"] - a_an) < 1e-4
    assert abs(res["b_star"] - b_an) < 1e-4

# ── 8. Comparación de métodos ────────────────────────────────────────────────
def t_comparacion_metodos():
    x, y = obtener_datos()
    r = reporte_analitico(x, y)
    res = descenso_gradiente(x, y, eta=ajustar_eta(x), max_iter=50_000)
    s_grad = error_cuadratico(x, y, res["a_star"], res["b_star"])
    assert s_grad >= r["s_min"] - 1e-9, "El gradiente no puede tener menor error que el mínimo analítico"

# ── 9. Métricas ──────────────────────────────────────────────────────────────
def t_metricas_ajuste_perfecto():
    x = np.array([1.0, 2.0, 3.0])
    y = 2.0 * x + 1.0
    y_hat = prediccion(x, 2.0, 1.0)
    assert abs(r_cuadrado(y, y_hat) - 1.0) < 1e-12
    assert mse(y, y_hat) < 1e-12 and rmse(y, y_hat) < 1e-12 and mae(y, y_hat) < 1e-12

def t_clasificacion_pendiente():
    assert "POSITIVA" in analizar_pendiente(1.5)
    assert "NEGATIVA" in analizar_pendiente(-2.0)
    assert "DÉBIL" in analizar_pendiente(0.0)

# ── 10. Exportación HyperFrames ──────────────────────────────────────────────
def t_export_hyperframes():
    res = exportar()
    js_path = os.path.join(RAIZ, "data", "generated", "datos.js")
    json_path = os.path.join(RAIZ, "data", "generated", "datos.json")
    assert os.path.exists(js_path) and os.path.exists(json_path)
    contenido = open(js_path, encoding="utf-8").read()
    assert contenido.startswith("window.HYPEFRAMES_DATA")
    datos_js = json.loads(contenido.replace("window.HYPEFRAMES_DATA = ", "").rstrip(";\n"))
    assert datos_js["total"] == res["total"]
    assert abs(datos_js["pendiente"] - res["pendiente"]) < 1e-12
    assert len(datos_js["datos"]) == res["total"]


if __name__ == "__main__":
    print("\n═══════ PRUEBAS AUTOMÁTICAS ═══════\n")
    prueba("1. Generador: reproducible y crea CSV", t_generador)
    prueba("1b. Generador: parámetros respetados (ruido=0)", t_generador_parametros)
    prueba("2. Carga y validación del CSV", t_carga_valida)
    prueba("2b. Validación rechaza x constantes", t_validacion_detecta_x_iguales)
    prueba("3. Función de error: S=0 en ajuste perfecto", t_error_cero_en_datos_perfectos)
    prueba("3b. Función de error: óptimo < otros puntos", t_error_propiedades)
    prueba("4. ∂S/∂a y ∂S/∂b vs diferencias finitas", t_gradiente_diferencias_finitas)
    prueba("4b. ∇S = 0 en el punto crítico", t_gradiente_cero_en_optimo)
    prueba("5. Fórmula cerrada = sistema lineal", t_ecuaciones_normales_consistentes)
    prueba("5b. Regresión recupera parámetros reales", t_recuperacion_parametros_reales)
    prueba("6. Hessiana simétrica y definida positiva", t_hessiana_simetrica_positiva)
    prueba("6b. Autovalores de H positivos (numpy)", t_hessiana_coincide_con_numpy)
    prueba("7. Gradiente converge al resultado analítico", t_gradiente_converge)
    prueba("8. Comparación: S_grad ≥ S_analítico", t_comparacion_metodos)
    prueba("9. Métricas: R²=1 y errores=0 (perfecto)", t_metricas_ajuste_perfecto)
    prueba("9b. Clasificación de pendiente", t_clasificacion_pendiente)
    prueba("10. Exportación a HyperFrames (js/json)", t_export_hyperframes)

    ok = sum(1 for _, ok, _ in resultados if ok)
    print(f"\n═══════ RESULTADO: {ok}/{len(resultados)} pruebas pasaron ═══════\n")
    sys.exit(0 if ok == len(resultados) else 1)
