"""
analisis.py
-----------
Análisis que pide el enunciado además del ajuste:

  §9   Comparación  Analítico (a mano) | Código propio | Biblioteca | Δ máx
  §8.7 Verificación NUMÉRICA del Hessiano (autovalores, diferencias finitas,
       número de condición)
  §11  Análisis de residuos (centrado, patrón, dispersión, atípicos, rachas)
  §12  Calidad del ajuste (RMSE, R², error relativo)
  §13  Análisis adicional:
         B — influencia de cada dato (se elimina uno y se reajusta)
         C — comparación con el modelo cuadrático y = a·x² + b·x + c
         A — sensibilidad: se perturba cada y_i en ±1 lb
         D — efecto del número de datos (subconjuntos de m = 3..n)
"""

import numpy as np

from analitico import resolver_analitico
from derivadas import hessiana
from metricas import r_cuadrado
from modelo import error_cuadratico, prediccion

# ─────────────────────────────────────────────────────────────────────────────
# Resultados del CÁLCULO MANUAL (docs/calculo_manual.md). Se escriben aquí a
# mano, tal como quedaron en el cuaderno, para compararlos con el código.
#   a* = 14490 / 4200            = 3.45
#   b* = (2051 − 3.45·975) / 15  = −1312.75 / 15 = −87.5167 (4 decimales)
#   S_mín = Σy² − a*·Σxy − b*·Σy = 30.2333 (4 decimales)
# ─────────────────────────────────────────────────────────────────────────────
MANUAL = {"a": 3.45, "b": -87.5167, "S": 30.2333}


# ═════════════════════════ §9 · COMPARACIÓN ═════════════════════════
def comparar_metodos(x, y, manual=MANUAL) -> dict:
    """Tabla de comparación entre cálculo manual, código propio y bibliotecas."""
    a_cod, b_cod = resolver_analitico(x, y)
    S_cod = error_cuadratico(x, y, a_cod, b_cod)

    # Biblioteca 1: numpy.polyfit (mínimos cuadrados polinomiales)
    a_np, b_np = np.polyfit(x, y, 1)
    S_np = error_cuadratico(x, y, a_np, b_np)

    lib = {"polyfit": (float(a_np), float(b_np), float(S_np))}

    # Biblioteca 2: numpy.linalg.lstsq sobre la matriz de diseño [x 1]
    (a_ls, b_ls), *_ = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)
    lib["lstsq"] = (float(a_ls), float(b_ls), error_cuadratico(x, y, a_ls, b_ls))

    # Biblioteca 3: scipy.stats.linregress (si está instalada)
    try:
        from scipy import stats
        r = stats.linregress(x, y)
        lib["linregress"] = (float(r.slope), float(r.intercept),
                             error_cuadratico(x, y, r.slope, r.intercept))
    except ImportError:
        pass

    a_lib, b_lib, S_lib = lib.get("linregress", lib["polyfit"])

    def dmax(*v):
        return float(max(v) - min(v))

    filas = [
        ("a", manual["a"], a_cod, a_lib, dmax(manual["a"], a_cod, a_lib), dmax(a_cod, a_lib)),
        ("b", manual["b"], b_cod, b_lib, dmax(manual["b"], b_cod, b_lib), dmax(b_cod, b_lib)),
        ("S_mín", manual["S"], S_cod, S_lib, dmax(manual["S"], S_cod, S_lib), dmax(S_cod, S_lib)),
    ]
    return {"filas": filas, "bibliotecas": lib}


def imprimir_comparacion(c: dict) -> None:
    print("\n" + "═" * 86)
    print("  §9  COMPARACIÓN: ANALÍTICO (MANUAL) vs CÓDIGO PROPIO vs BIBLIOTECA (scipy/numpy)")
    print("═" * 86)
    print(f"  {'Parámetro':<10}{'Analítico':>14}{'Código propio':>18}{'Biblioteca':>18}"
          f"{'Δ máx (3)':>13}{'Δ cód–lib':>13}")
    for nombre, m, cod, lib, d3, d2 in c["filas"]:
        print(f"  {nombre:<10}{m:>14.4f}{cod:>18.10f}{lib:>18.10f}{d3:>13.2e}{d2:>13.2e}")
    print("\n  • La diferencia manual–código (~1e-5) viene solo del REDONDEO a 4 decimales")
    print("    hecho a mano. Código propio vs biblioteca difieren ~1e-13: error de punto")
    print("    flotante. Los tres métodos encuentran el mismo mínimo.")
    for k, (a, b, S) in c["bibliotecas"].items():
        print(f"  • {k:<11}: a = {a:.10f}   b = {b:.10f}   S = {S:.10f}")
    print("═" * 86)


# ═════════════════════ §8.7 · HESSIANO NUMÉRICO ═════════════════════
def verificar_hessiano_numerico(x, y, a, b, h=1e-3) -> dict:
    """
    Comprueba numéricamente las propiedades del Hessiano:
      • H por diferencias finitas centradas coincide con la fórmula analítica
      • H es simétrica
      • sus autovalores son positivos  → definida positiva → mínimo
      • número de condición κ = λ_max / λ_min (forma de las curvas de nivel)
      • H no depende de (a, b): se evalúa en otro punto y da lo mismo
    """
    def S(p, q):
        return error_cuadratico(x, y, p, q)

    def H_num(p, q):
        Saa = (S(p + h, q) - 2 * S(p, q) + S(p - h, q)) / h ** 2
        Sbb = (S(p, q + h) - 2 * S(p, q) + S(p, q - h)) / h ** 2
        Sab = (S(p + h, q + h) - S(p + h, q - h) - S(p - h, q + h) + S(p - h, q - h)) / (4 * h ** 2)
        return np.array([[Saa, Sab], [Sab, Sbb]])

    H = hessiana(x)
    Hn = H_num(a, b)
    Hn_otro = H_num(0.0, 0.0)
    eig = np.linalg.eigvalsh(H)
    _, vecs = np.linalg.eigh(H)
    return {
        "H": H,
        "H_num": Hn,
        "H_num_otro": Hn_otro,
        "error_rel": float(np.max(np.abs(Hn - H)) / np.max(np.abs(H))),
        "simetrica": bool(np.allclose(H, H.T)),
        "autovalores": eig,
        "autovectores": vecs,
        "definida_positiva": bool(np.all(eig > 0)),
        "det": float(np.linalg.det(H)),
        "traza": float(np.trace(H)),
        "condicion": float(eig[-1] / eig[0]),
    }


def imprimir_hessiano_numerico(v: dict) -> None:
    H, Hn, eig = v["H"], v["H_num"], v["autovalores"]
    print("\n" + "═" * 64)
    print("  §8.6–8.7  HESSIANO: CÁLCULO Y VERIFICACIÓN NUMÉRICA")
    print("═" * 64)
    print(f"  H (fórmula)          = [[{H[0,0]:.2f}, {H[0,1]:.2f}], [{H[1,0]:.2f}, {H[1,1]:.2f}]]")
    print(f"  H (dif. finitas)     = [[{Hn[0,0]:.2f}, {Hn[0,1]:.2f}], [{Hn[1,0]:.2f}, {Hn[1,1]:.2f}]]")
    print(f"  Error relativo       = {v['error_rel']:.2e}")
    print(f"  Simétrica            = {v['simetrica']}")
    print(f"  det(H)               = {v['det']:.4f}  (> 0)")
    print(f"  ∂²S/∂a²              = {H[0,0]:.2f}  (> 0)")
    print(f"  Autovalores λ₁, λ₂   = {eig[0]:.6f}, {eig[1]:.2f}  (ambos > 0)")
    print(f"  → Definida positiva  = {v['definida_positiva']}  ⇒  (a*, b*) es MÍNIMO (global, S es cuadrática)")
    print(f"  Número de condición  κ = λ₂/λ₁ = {v['condicion']:.3e}")
    print("    κ enorme ⇒ las curvas de nivel son elipses MUY alargadas: S cambia")
    print("    mucho en una dirección y casi nada en la otra (valle largo y estrecho).")
    print(f"  H evaluado en (0,0)  = [[{v['H_num_otro'][0,0]:.2f}, ...]] → H es constante")
    print("═" * 64)


# ═════════════════════ §11–12 · RESIDUOS Y CALIDAD ═════════════════════
def _rachas(signos):
    """Número de rachas (cambios de signo + 1)."""
    s = [v for v in signos if v != 0]
    return 1 + sum(1 for i in range(1, len(s)) if s[i] != s[i - 1]) if s else 0


def analizar_residuos(x, y, a, b) -> dict:
    # Rachas, mitades y tercios solo tienen sentido con los x ORDENADOS;
    # el CSV puede venir en cualquier orden.
    orden = np.argsort(x, kind="stable")
    x, y = np.asarray(x, float)[orden], np.asarray(y, float)[orden]
    e = y - prediccion(x, a, b)
    n = len(e)
    s = float(np.sqrt(np.sum(e ** 2) / (n - 2)))           # error estándar de la regresión
    z = e / s
    signos = np.sign(e).astype(int)
    mitad = n // 2
    # patrón curvo: correlación de e con (x − x̄)²
    u2 = (x - x.mean()) ** 2
    corr_curva = float(np.corrcoef(u2, e)[0, 1])
    # cambio de dispersión: |e| en la primera vs segunda mitad
    disp_1 = float(np.mean(np.abs(e[:mitad])))
    disp_2 = float(np.mean(np.abs(e[mitad:])))
    tercio = n // 3
    regiones = {
        "extremo izquierdo": float(np.mean(e[:tercio])),
        "centro": float(np.mean(e[tercio:n - tercio])),
        "extremo derecho": float(np.mean(e[n - tercio:])),
    }
    return {
        "e": e,
        "media": float(np.mean(e)),
        "suma": float(np.sum(e)),
        "suma_x_e": float(np.sum(x * e)),
        "s": s,
        "z": z,
        "atipicos": [(float(x[i]), float(e[i]), float(z[i])) for i in range(n) if abs(z[i]) > 2],
        "rachas": _rachas(signos),
        "positivos": int(np.sum(e > 0)),
        "negativos": int(np.sum(e < 0)),
        "corr_curvatura": corr_curva,
        "disp_1": disp_1,
        "disp_2": disp_2,
        "regiones": regiones,
        "max_pos": (float(x[np.argmax(e)]), float(np.max(e))),
        "max_neg": (float(x[np.argmin(e)]), float(np.min(e))),
    }


def calidad_ajuste(x, y, a, b) -> dict:
    yh = prediccion(x, a, b)
    e = y - yh
    n = len(y)
    ss_res = float(np.sum(e ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    rmse = float(np.sqrt(ss_res / n))
    return {
        "SS_res": ss_res, "SS_tot": ss_tot,
        "R2": r_cuadrado(y, yh),
        "RMSE": rmse,
        "MAE": float(np.mean(np.abs(e))),
        "rango_y": float(y.max() - y.min()),
        "desv_y": float(np.std(y)),
        "rmse_rel_media": rmse / float(y.mean()),
        "rmse_rel_rango": rmse / float(y.max() - y.min()),
    }


def imprimir_residuos_calidad(r: dict, q: dict, ux="in", uy="lb") -> None:
    print("\n" + "═" * 64)
    print("  §11  ANÁLISIS DE RESIDUOS")
    print("═" * 64)
    print(f"  Media de e_i        = {r['media']:.2e}   (Σe_i = {r['suma']:.1e}): centrados en 0")
    print(f"  Σ x_i·e_i           = {r['suma_x_e']:.1e}   (son exactamente las condiciones ∇S = 0)")
    print(f"  e_i > 0: {r['positivos']}   e_i < 0: {r['negativos']}   rachas de signo: {r['rachas']}")
    for k, v in r["regiones"].items():
        print(f"  Residuo medio en {k:<18}: {v:+.3f} {uy}")
    print(f"  corr(e, (x−x̄)²)      = {r['corr_curvatura']:.3f}  → patrón CURVO (forma de U)" if abs(r['corr_curvatura']) > 0.5
          else f"  corr(e, (x−x̄)²)      = {r['corr_curvatura']:.3f}")
    print(f"  |e| medio 1ª mitad  = {r['disp_1']:.3f}   2ª mitad = {r['disp_2']:.3f}")
    print(f"  Mayor residuo +     : e = {r['max_pos'][1]:+.3f} {uy} en x = {r['max_pos'][0]:.0f} {ux}")
    print(f"  Mayor residuo −     : e = {r['max_neg'][1]:+.3f} {uy} en x = {r['max_neg'][0]:.0f} {ux}")
    print(f"  Atípicos (|e/s|>2)  : {[(xx, round(ee, 3)) for xx, ee, _ in r['atipicos']] or 'ninguno'}")

    print("\n" + "═" * 64)
    print("  §12  CALIDAD DEL AJUSTE")
    print("═" * 64)
    print(f"  R²    = {q['R2']:.6f}   (la recta explica el {100*q['R2']:.2f}% de la variación de y)")
    print(f"  RMSE  = {q['RMSE']:.4f} {uy}   ({100*q['rmse_rel_media']:.2f}% del peso medio, "
          f"{100*q['rmse_rel_rango']:.2f}% del rango de {q['rango_y']:.0f} {uy})")
    print(f"  MAE   = {q['MAE']:.4f} {uy}")
    print("═" * 64)


# ═════════════════════ §13 · ANÁLISIS ADICIONAL ═════════════════════
def influencia_datos(x, y) -> dict:
    """Opción B: elimina cada punto, reajusta y mide cuánto cambia la recta."""
    a0, b0 = resolver_analitico(x, y)
    filas = []
    for i in range(len(x)):
        m = np.ones(len(x), bool); m[i] = False
        a_i, b_i = resolver_analitico(x[m], y[m])
        # cambio de la predicción en todo el rango (distancia de Cook simplificada)
        cambio_pred = float(np.max(np.abs(prediccion(x, a_i, b_i) - prediccion(x, a0, b0))))
        filas.append({"i": i + 1, "x": float(x[i]), "y": float(y[i]),
                      "a": a_i, "b": b_i, "da": a_i - a0, "db": b_i - b0,
                      "cambio_pred": cambio_pred})
    mas = max(filas, key=lambda f: abs(f["da"]))
    return {"a0": a0, "b0": b0, "filas": filas, "mas_influyente": mas}


def ajuste_cuadratico_propio(x, y) -> np.ndarray:
    """
    Opción C: ajuste y = p·x² + q·x + r minimizando
        S(p,q,r) = Σ (y_i − p·x_i² − q·x_i − r)²
    ∇S = 0 da el sistema 3×3 de ecuaciones normales:
        | Σx⁴ Σx³ Σx² | |p|   |Σx²y|
        | Σx³ Σx² Σx  | |q| = |Σxy |
        | Σx² Σx  n   | |r|   |Σy  |
    """
    M = np.array([[np.sum(x**4), np.sum(x**3), np.sum(x**2)],
                  [np.sum(x**3), np.sum(x**2), np.sum(x)],
                  [np.sum(x**2), np.sum(x),    len(x)]], float)
    v = np.array([np.sum(x**2 * y), np.sum(x * y), np.sum(y)], float)
    return np.linalg.solve(M, v)


def comparar_cuadratico(x, y) -> dict:
    n = len(x)
    # 3 parámetros + la prueba F usa n − 3 grados de libertad → n ≥ 4
    if n < 4 or len(np.unique(x)) < 3:
        raise ValueError("El modelo cuadrático necesita n ≥ 4 y al menos 3 valores de x distintos.")
    a, b = resolver_analitico(x, y)
    p, q, r = ajuste_cuadratico_propio(x, y)
    p_np, q_np, r_np = np.polyfit(x, y, 2)
    yl = prediccion(x, a, b)
    yc = p * x**2 + q * x + r
    S_l = float(np.sum((y - yl) ** 2))
    S_c = float(np.sum((y - yc) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    # Prueba F parcial: ¿el término x² mejora significativamente?
    F = (S_l - S_c) / (S_c / (n - 3))
    try:
        from scipy import stats
        p_valor = float(stats.f.sf(F, 1, n - 3))
    except ImportError:
        p_valor = float("nan")
    return {
        "lineal": (a, b), "cuadratico": (p, q, r),
        "cuadratico_np": (p_np, q_np, r_np),
        "S_lineal": S_l, "S_cuad": S_c,
        "R2_lineal": 1 - S_l / ss_tot, "R2_cuad": 1 - S_c / ss_tot,
        "RMSE_lineal": float(np.sqrt(S_l / n)), "RMSE_cuad": float(np.sqrt(S_c / n)),
        "max_dif_pred": float(np.max(np.abs(yc - yl))),
        "F": float(F), "p_valor": p_valor,
        "e_cuad": y - yc,
    }


def sensibilidad(x, y, delta=1.0) -> dict:
    """Opción A: suma ±delta a cada y_i (uno por uno) y mide el cambio en a*, b*."""
    a0, b0 = resolver_analitico(x, y)
    da, db = [], []
    for i in range(len(y)):
        yy = y.copy(); yy[i] += delta
        a1, b1 = resolver_analitico(x, yy)
        da.append(a1 - a0); db.append(b1 - b0)
    return {"da": np.array(da), "db": np.array(db), "delta": delta}


def imprimir_adicional(inf: dict, cq: dict, sen: dict, ux="in", uy="lb") -> None:
    print("\n" + "═" * 72)
    print("  §13-B  INFLUENCIA DE UN DATO (se elimina cada punto y se reajusta)")
    print("═" * 72)
    print(f"  {'i':>3} {'x':>5} {'y':>6} {'a sin i':>10} {'Δa':>9} {'b sin i':>10} {'Δb':>9} {'máx Δŷ':>8}")
    for f in inf["filas"]:
        print(f"  {f['i']:>3} {f['x']:>5.0f} {f['y']:>6.0f} {f['a']:>10.4f} {f['da']:>+9.4f} "
              f"{f['b']:>10.3f} {f['db']:>+9.3f} {f['cambio_pred']:>8.3f}")
    m = inf["mas_influyente"]
    print(f"  → Punto más influyente: x = {m['x']:.0f} {ux}, y = {m['y']:.0f} {uy} "
          f"(Δa = {m['da']:+.4f}, Δb = {m['db']:+.3f})")

    print("\n" + "═" * 72)
    print("  §13-C  LINEAL vs CUADRÁTICO")
    print("═" * 72)
    p, q, r = cq["cuadratico"]
    print(f"  Lineal     : ŷ = {cq['lineal'][0]:.4f}·x + ({cq['lineal'][1]:.4f})")
    print(f"  Cuadrático : ŷ = {p:.6f}·x² + ({q:.4f})·x + {r:.3f}   (código propio, sistema 3×3)")
    print(f"               polyfit grado 2 → {cq['cuadratico_np'][0]:.6f}, {cq['cuadratico_np'][1]:.4f}, {cq['cuadratico_np'][2]:.3f}")
    print(f"  S     : lineal = {cq['S_lineal']:.4f}   cuadrático = {cq['S_cuad']:.4f}")
    print(f"  R²    : lineal = {cq['R2_lineal']:.6f}   cuadrático = {cq['R2_cuad']:.6f}")
    print(f"  RMSE  : lineal = {cq['RMSE_lineal']:.4f} {uy}   cuadrático = {cq['RMSE_cuad']:.4f} {uy}")
    print(f"  Máx. diferencia entre predicciones = {cq['max_dif_pred']:.3f} {uy}")
    print(f"  Prueba F parcial (término x²): F = {cq['F']:.1f}, p = {cq['p_valor']:.1e}")

    print("\n" + "═" * 72)
    print(f"  §13-A  SENSIBILIDAD (y_i + {sen['delta']:.0f} {uy}, un punto a la vez)")
    print("═" * 72)
    print(f"  |Δa| máx = {np.max(np.abs(sen['da'])):.4f} {uy}/{ux}   |Δb| máx = {np.max(np.abs(sen['db'])):.4f} {uy}")
    print("  Δa es lineal en Δy (a* = Σ w_i·y_i): los puntos de los EXTREMOS pesan más,")
    print("  los del centro casi nada → la pendiente es poco sensible a un error de 1 lb.")
    print("═" * 72)


def efecto_n(x, y, repeticiones=200, semilla=0) -> dict:
    """
    Opción D: reajusta con m = 3..n datos y mide cómo cambian a*, b*.

    Para cada m se toman `repeticiones` subconjuntos aleatorios de m puntos
    (semilla fija → reproducible) y se reporta la media y la desviación de a*, b*.
    La desviación debe bajar a medida que m crece.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    rng = np.random.default_rng(semilla)
    filas = []
    for m in range(3, len(x) + 1):
        a_s, b_s = [], []
        for _ in range(repeticiones):
            idx = rng.choice(len(x), size=m, replace=False)
            if np.ptp(x[idx]) == 0:
                continue    # todos los x iguales: no hay recta única
            a_i, b_i = resolver_analitico(x[idx], y[idx])
            a_s.append(a_i); b_s.append(b_i)
        filas.append({"m": m, "a_media": float(np.mean(a_s)), "a_desv": float(np.std(a_s)),
                      "b_media": float(np.mean(b_s)), "b_desv": float(np.std(b_s))})
    return {"filas": filas, "repeticiones": repeticiones}


def imprimir_efecto_n(d: dict, ux="in", uy="lb") -> None:
    print("\n" + "═" * 72)
    print(f"  §13-D  EFECTO DEL NÚMERO DE DATOS ({d['repeticiones']} subconjuntos aleatorios por m)")
    print("═" * 72)
    print(f"  {'m':>3} {'a* medio':>10} {'desv a*':>9} {'b* medio':>10} {'desv b*':>9}")
    for f in d["filas"]:
        print(f"  {f['m']:>3} {f['a_media']:>10.4f} {f['a_desv']:>9.4f} "
              f"{f['b_media']:>10.3f} {f['b_desv']:>9.3f}")
    print("  → Con más datos la dispersión de a* y b* disminuye: el ajuste se estabiliza.")
    print("═" * 72)
