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
 10. Exportación a HyperFrames (archivos generados por main.py)
 11. Análisis: comparación con bibliotecas, Hessiano numérico, Newton,
     modelo cuadrático propio
"""

import os
import sys
import json
import tempfile
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
from gradiente import descenso_gradiente, ajustar_eta, metodo_newton
from metricas import r_cuadrado, mse, rmse, mae, analizar_pendiente
import analisis

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
    # se escribe en una carpeta temporal: NUNCA se tocan los datos reales
    ruta = guardar_csv(x1, y1, os.path.join(tempfile.mkdtemp(), "sint.csv"))
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
        # mismas longitudes y n = 3: el único defecto es que x es constante
        validar_datos(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))
        assert False, "Debía rechazar x constantes"
    except ValueError as ex:
        assert "iguales" in str(ex), f"Rechazó por otro motivo: {ex}"

def t_validacion_n_minimo():
    try:
        validar_datos(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
        assert False, "Debía rechazar n = 2"
    except ValueError as ex:
        assert "3 observaciones" in str(ex)

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
    # Con los datos reales el Hessiano está mal condicionado (κ ≈ 1e6):
    # se usa la versión con x centrada, que sí converge.
    x, y = obtener_datos()
    res = descenso_gradiente(x, y, centrar=True, max_iter=50_000)
    assert res["convergio"], "El descenso no convergió"
    a_an, b_an = resolver_analitico(x, y)
    assert abs(res["a_star"] - a_an) < 1e-4
    assert abs(res["b_star"] - b_an) < 1e-4

# ── 8. Comparación de métodos ────────────────────────────────────────────────
def t_comparacion_metodos():
    x, y = obtener_datos()
    r = reporte_analitico(x, y)
    res = descenso_gradiente(x, y, max_iter=5_000)
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
    # Genera la exportación en una carpeta temporal: no depende de haber
    # ejecutado main.py antes ni modifica data/generated/.
    import main
    from metricas import imprimir_metricas
    x, y = obtener_datos()
    r = reporte_analitico(x, y)
    a, b = r["a_star"], r["b_star"]
    g = descenso_gradiente(x, y, centrar=True, max_iter=20_000)
    original = main.CARPETA_HYPERFRAMES
    with tempfile.TemporaryDirectory() as tmp:
        main.CARPETA_HYPERFRAMES = tmp
        try:
            main.exportar_hyperframes(x, y, r, verificar_minimo(x), g,
                                      imprimir_metricas(x, y, a, b))
        finally:
            main.CARPETA_HYPERFRAMES = original
        contenido = open(os.path.join(tmp, "datos.js"), encoding="utf-8").read()
    assert contenido.startswith("window.HYPEFRAMES_DATA")
    datos_js = json.loads(contenido.replace("window.HYPEFRAMES_DATA = ", "").rstrip(";\n"))
    assert datos_js["total"] == len(x)
    assert abs(datos_js["pendiente"] - a) < 1e-9
    assert len(datos_js["datos"]) == len(x)

# ── 11. Análisis ─────────────────────────────────────────────────────────────
def t_comparacion_bibliotecas():
    x, y = obtener_datos()
    c = analisis.comparar_metodos(x, y)
    for nombre, manual, cod, lib, d3, d2 in c["filas"]:
        assert d2 < 1e-8, f"{nombre}: código propio y biblioteca difieren {d2}"
        assert d3 < 1e-3, f"{nombre}: el valor manual difiere {d3} (¿error a mano?)"

def t_hessiano_numerico():
    x, y = obtener_datos()
    a, b = resolver_analitico(x, y)
    v = analisis.verificar_hessiano_numerico(x, y, a, b)
    assert v["error_rel"] < 1e-5 and v["definida_positiva"] and v["simetrica"]

def t_newton_un_paso():
    x, y = obtener_datos()
    a, b = resolver_analitico(x, y)
    nw = metodo_newton(x, y, 10.0, 10.0)
    assert abs(nw["a_star"] - a) < 1e-9 and abs(nw["b_star"] - b) < 1e-6

def t_cuadratico_propio():
    x, y = obtener_datos()
    p = analisis.ajuste_cuadratico_propio(x, y)
    assert np.allclose(p, np.polyfit(x, y, 2), rtol=1e-6)
    cq = analisis.comparar_cuadratico(x, y)
    assert cq["S_cuad"] <= cq["S_lineal"]

# ── 12. Robustez (F1) ────────────────────────────────────────────────────────
def t_residuos_independientes_del_orden():
    x, y = obtener_datos()
    a, b = resolver_analitico(x, y)
    p = np.random.default_rng(1).permutation(len(x))
    r1 = analisis.analizar_residuos(x, y, a, b)
    r2 = analisis.analizar_residuos(x[p], y[p], a, b)
    assert r1["rachas"] == r2["rachas"] == 3, (r1["rachas"], r2["rachas"])
    assert r1["regiones"] == r2["regiones"] and r1["disp_1"] == r2["disp_1"]

def t_r2_indefinido():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isnan(r_cuadrado(np.array([2.0, 2.0, 2.0]), np.array([2.0, 2.0, 2.0])))

def t_lstsq_en_bibliotecas():
    x, y = obtener_datos()
    a, b = resolver_analitico(x, y)
    a_ls, b_ls, _ = analisis.comparar_metodos(x, y)["bibliotecas"]["lstsq"]
    assert abs(a_ls - a) < 1e-9 and abs(b_ls - b) < 1e-6

def t_cuadratico_rechaza_n_pequeno():
    try:
        analisis.comparar_cuadratico(np.array([1.0, 2.0, 3.0]), np.array([1.0, 4.0, 9.0]))
        assert False, "Debía rechazar n = 3"
    except ValueError:
        pass

def t_efecto_n():
    x, y = obtener_datos()
    a, b = resolver_analitico(x, y)
    filas = analisis.efecto_n(x, y, repeticiones=50)["filas"]
    assert [f["m"] for f in filas] == list(range(3, len(x) + 1))
    # con m = n todos los subconjuntos son el conjunto completo
    assert abs(filas[-1]["a_media"] - a) < 1e-9 and filas[-1]["a_desv"] < 1e-9
    assert filas[0]["a_desv"] > filas[-2]["a_desv"], "La dispersión debe bajar al crecer m"


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
    prueba("11. Manual = código propio = biblioteca", t_comparacion_bibliotecas)
    prueba("11b. Hessiano numérico = analítico, def. positivo", t_hessiano_numerico)
    prueba("11c. Newton llega al mínimo en 1 paso", t_newton_un_paso)
    prueba("11d. Ajuste cuadrático propio = polyfit", t_cuadratico_propio)
    prueba("2c. Validación rechaza n < 3", t_validacion_n_minimo)
    prueba("12. Residuos no dependen del orden del CSV", t_residuos_independientes_del_orden)
    prueba("12b. R² = nan si todos los y son iguales", t_r2_indefinido)
    prueba("12c. numpy.linalg.lstsq = código propio", t_lstsq_en_bibliotecas)
    prueba("12d. Cuadrático rechaza n < 4", t_cuadratico_rechaza_n_pequeno)
    prueba("12e. Opción D: efecto del número de datos", t_efecto_n)

    ok = sum(1 for _, ok, _ in resultados if ok)
    print(f"\n═══════ RESULTADO: {ok}/{len(resultados)} pruebas pasaron ═══════\n")
    sys.exit(0 if ok == len(resultados) else 1)
