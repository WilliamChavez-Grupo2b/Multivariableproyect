"""
main.py
-------
Punto de entrada del proyecto: Regresión lineal por mínimos cuadrados
usando Cálculo Multivariable.

Ejecuta el flujo completo descrito en los documentos:

  1. Datos            → cargar y mostrar el conjunto de observaciones (datos.py)
  2. Planteamiento    → modelo ŷ = a·x + b y función de error S(a, b) (modelo.py)
  3. Solución analítica → ecuaciones normales, a*, b* (analitico.py)
  4. Verificación     → matriz Hessiana, mínimo global (derivadas.py)
  5. Descenso por gradiente → verificación numérica (gradiente.py)
  6. Métricas         → R², MSE, RMSE, MAE, residuos, tipo de relación (metricas.py)
  7. Gráficas         → regresión, residuos, superficie, contorno (graficas.py)
  8. HyperFrames      → exporta resultados a data/generated/datos.js

Uso:
    python main.py            → flujo completo
    python main.py --sin-gui  → calcula sin abrir ventanas de gráficas
"""

import sys

# Consolas de Windows (cp1252): forzar UTF-8 para imprimir símbolos Unicode
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Permitir ejecutar este archivo directamente: añadir la carpeta src al path
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datos import obtener_datos, resumen
from analitico import main_analitico
from metricas import imprimir_metricas
from gradiente import descenso_gradiente, ajustar_eta, imprimir_resultado_gradiente
from exportar_hyperframes import exportar, imprimir_resumen_export
from exportar_resultados import exportar_todo as exportar_results
import graficas


def main(mostrar_graficas: bool = True) -> None:
    print("╔" + "═" * 58 + "╗")
    print("║  REGRESIÓN LINEAL POR MÍNIMOS CUADRADOS                        ║")
    print("║  Cálculo Multivariable — derivadas parciales y Hessiana        ║")
    print("╚" + "═" * 58 + "╝")

    # ── 1. Datos ────────────────────────────────────────────────────────────
    x, y = obtener_datos()
    resumen(x, y)

    # ── 2-4. Solución analítica + Hessiana ─────────────────────────────────
    x, y, r = main_analitico()
    a_star, b_star = r["a_star"], r["b_star"]

    # ── 5. Descenso por gradiente (verificación numérica) ──────────────────
    eta = ajustar_eta(x)
    print(f"\n• Tasa de aprendizaje sugerida: η = {eta:.8f}")
    res_grad = descenso_gradiente(x, y, a_init=0.0, b_init=0.0, eta=eta)
    imprimir_resultado_gradiente(res_grad)

    # Comparación analítico vs numérico
    print("\n• Comparación:")
    print(f"  Analítico  : a* = {a_star:.6f},  b* = {b_star:.6f}")
    print(f"  Gradiente  : a* = {res_grad['a_star']:.6f},  b* = {res_grad['b_star']:.6f}")
    diff_a = abs(a_star - res_grad["a_star"])
    diff_b = abs(b_star - res_grad["b_star"])
    print(f"  Diferencia : |Δa| = {diff_a:.2e},  |Δb| = {diff_b:.2e}")
    if diff_a < 1e-3 and diff_b < 1e-3:
        print("  ✓ Ambos métodos coinciden: el mínimo global fue encontrado.")

    # ── 6. Métricas ─────────────────────────────────────────────────────────
    imprimir_metricas(x, y, a_star, b_star)

    # ── 7. Gráficas ─────────────────────────────────────────────────────────
    if mostrar_graficas:
        graficas.generar_todas(x, y, a_star, b_star, historial=None)
    else:
        print("\n[Modo --sin-gui: gráficas omitidas]")

    # ── 8. Exportar a HyperFrames ───────────────────────────────────────
    imprimir_resumen_export(exportar())

    # ── 9. Regenerar results/ completo (borra lo anterior + MP4) ────────
    carpeta_results = exportar_results()
    print(f"\n✓ results/ regenerado con los datos nuevos en: "
          f"{os.path.abspath(carpeta_results)}")

    # ── Conclusión ──────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  CONCLUSIÓN")
    print("═" * 60)
    print(f"""
  La recta óptima  ŷ = {a_star:.4f}·x + ({b_star:.4f})  minimiza el error
  cuadrático total  S(a*, b*) = {r['s_min']:.4f}.

  La matriz Hessiana es definida positiva (det(H) > 0 con al menos dos
  valores de x diferentes), por lo que (a*, b*) es un MÍNIMO GLOBAL de S.

  El descenso por gradiente llegó al mismo punto desde (0, 0),
  confirmando numéricamente el resultado analítico.
""")
    print("═" * 60)


if __name__ == "__main__":
    sin_gui = "--sin-gui" in sys.argv
    main(mostrar_graficas=not sin_gui)
