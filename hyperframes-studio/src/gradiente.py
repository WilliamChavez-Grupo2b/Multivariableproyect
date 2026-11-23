"""
gradiente.py
------------
Métodos NUMÉRICOS (iterativos) para minimizar S(a, b). Son un complemento:
la solución oficial del proyecto es la ANALÍTICA (ecuaciones normales).

El gradiente de S(a, b) es (ver docs/desarrollo_matematico.md):
  ∂S/∂a = -2 · Σ x_i · (y_i - a·x_i - b)  =  2·(a·Σx² + b·Σx − Σxy)
  ∂S/∂b = -2 · Σ (y_i - a·x_i - b)        =  2·(a·Σx  + n·b  − Σy)

Se implementan tres métodos:

1. Descenso por gradiente "simple" en (a, b):
       (a, b) ← (a, b) − η·∇S
   Con estatura entre 58 y 72 in, las curvas de nivel de S son elipses MUY
   alargadas: el Hessiano tiene número de condición κ = λ_max/λ_min ≈ 10⁶.
   El método avanza rapidísimo en la dirección empinada y casi nada en la
   dirección plana → NO converge en un número razonable de iteraciones.
   Esto es un resultado importante: la FORMA geométrica de S (dada por el
   Hessiano) determina qué tan fácil es encontrar el mínimo numéricamente.

2. Descenso por gradiente con x CENTRADA  (u_i = x_i − x̄):
       y = a·u + c,   con   b = c − a·x̄
   En estas variables el Hessiano es diagonal:  H = diag(2·Σu², 2n)
   (porque Σu = 0) → las elipses quedan alineadas con los ejes, κ ≈ 19,
   y el método converge en unos cientos de iteraciones.

3. Método de Newton (usa el Hessiano):
       (a, b) ← (a, b) − H⁻¹·∇S
   Como S es cuadrática, Newton llega al mínimo EXACTO en UNA iteración.
"""

import numpy as np

from modelo import error_cuadratico


def _sumas(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return (len(x), float(x.sum()), float(y.sum()),
            float((x * y).sum()), float((x * x).sum()))


def _registrar(it, max_iter):
    """Guarda todas las primeras 300 iteraciones y luego ~400 puntos log-espaciados."""
    if it <= 300:
        return True
    paso = max(1, max_iter // 400)
    return it % paso == 0


def descenso_gradiente(
    x: np.ndarray,
    y: np.ndarray,
    a_init: float = 0.0,
    b_init: float = 0.0,
    eta: float | None = None,
    max_iter: int = 200_000,
    tolerancia: float = 1e-6,
    centrar: bool = False,
) -> dict:
    """
    Minimiza S(a, b) por descenso de gradiente.

    centrar=False → iteración directa sobre (a, b).
    centrar=True  → iteración sobre (a, c) con u = x − x̄, y luego b = c − a·x̄.

    Retorna dict con a_star, b_star, iteraciones, convergio, historial y
    trayectoria = lista de (a, b, S(a,b), ‖∇S‖, iteración) en las variables
    ORIGINALES (a, b), para poder dibujarla sobre las curvas de nivel de S.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    x_mean = float(x.mean())
    xx = x - x_mean if centrar else x

    n, sx, sy, sxy, sxx = _sumas(xx, y)
    if eta is None:
        eta = ajustar_eta(xx)

    a = float(a_init)
    # en variables centradas, el intercepto equivalente de (a_init, b_init) es c = b + a·x̄
    p = float(b_init) + (a * x_mean if centrar else 0.0)

    def grad(a, p):
        return 2.0 * (a * sxx + p * sx - sxy), 2.0 * (a * sx + n * p - sy)

    def a_original(a, p):
        return a, (p - a * x_mean) if centrar else p

    def S_y_norma(a, p):
        ga, gp = grad(a, p)
        ao, bo = a_original(a, p)
        # el gradiente se reporta en variables originales (a, b)
        gA = ga - gp * x_mean if centrar else ga
        return error_cuadratico(x, y, ao, bo), float(np.hypot(gA, gp))

    S0, g0 = S_y_norma(a, p)
    trayectoria = [(*a_original(a, p), S0, g0, 0)]
    historial = [(0, S0, g0)]
    convergio = False
    it = 0

    for it in range(1, max_iter + 1):
        ga, gp = grad(a, p)
        a -= eta * ga
        p -= eta * gp

        if _registrar(it, max_iter):
            S, g = S_y_norma(a, p)
            trayectoria.append((*a_original(a, p), S, g, it))
            if it % 500 == 0 or it == 1:
                historial.append((it, S, g))
            if g < tolerancia:
                convergio = True
                break
        elif it % 50 == 0:
            _, g = S_y_norma(a, p)
            if g < tolerancia:
                S, g = S_y_norma(a, p)
                trayectoria.append((*a_original(a, p), S, g, it))
                convergio = True
                break

    S, g = S_y_norma(a, p)
    if trayectoria[-1][4] != it:
        trayectoria.append((*a_original(a, p), S, g, it))
    historial.append((it, S, g))
    a_fin, b_fin = a_original(a, p)

    return {
        "a_star": a_fin,
        "b_star": b_fin,
        "iteraciones": it,
        "historial": historial,
        "trayectoria": trayectoria,
        "convergio": convergio,
        "eta": eta,
        "centrado": centrar,
        "norma_final": g,
    }


def metodo_newton(x: np.ndarray, y: np.ndarray,
                  a_init: float = 0.0, b_init: float = 0.0) -> dict:
    """
    Un paso de Newton:  (a, b)₁ = (a, b)₀ − H⁻¹ ∇S(a₀, b₀).
    Como S es un polinomio de grado 2, su aproximación de Taylor de orden 2
    es EXACTA y el paso cae justo en el mínimo.
    """
    n, sx, sy, sxy, sxx = _sumas(x, y)
    H = np.array([[2 * sxx, 2 * sx], [2 * sx, 2 * n]])
    g = np.array([2 * (a_init * sxx + b_init * sx - sxy),
                  2 * (a_init * sx + n * b_init - sy)])
    paso = np.linalg.solve(H, g)
    a1, b1 = a_init - paso[0], b_init - paso[1]
    g1 = np.array([2 * (a1 * sxx + b1 * sx - sxy), 2 * (a1 * sx + n * b1 - sy)])
    return {"a_star": float(a1), "b_star": float(b1), "iteraciones": 1,
            "norma_final": float(np.linalg.norm(g1))}


def ajustar_eta(x: np.ndarray) -> float:
    """
    Tasa de aprendizaje segura: η = 1 / (traza de H) = 1 / (2·Σx² + 2n).
    Como traza(H) = λ₁ + λ₂ ≥ λ_max, se cumple η < 2/λ_max (condición de
    convergencia del descenso por gradiente en una función cuadrática).
    """
    x = np.asarray(x, float)
    return 1.0 / (2.0 * float(np.sum(x ** 2)) + 2.0 * len(x))


def imprimir_resultado_gradiente(res: dict, titulo: str = "DESCENSO POR GRADIENTE") -> None:
    """Imprime el resultado del descenso por gradiente."""
    print("\n" + "═" * 55)
    print(f"  {titulo}")
    print("═" * 55)
    print(f"  η = {res['eta']:.3e}")
    print(f"\n  {'Iter':>8}  {'S(a,b)':>14}  {'‖∇S‖':>14}")
    print(f"  {'─'*8}  {'─'*14}  {'─'*14}")
    for (it, S, norma) in res["historial"][-6:]:
        print(f"  {it:>8}  {S:>14.6f}  {norma:>14.6e}")
    print(f"\n• Iteraciones totales : {res['iteraciones']}")
    print(f"• Convergió (‖∇S‖<1e-6): {res['convergio']}")
    print(f"• a = {res['a_star']:.6f}   b = {res['b_star']:.6f}")
    print("═" * 55)
