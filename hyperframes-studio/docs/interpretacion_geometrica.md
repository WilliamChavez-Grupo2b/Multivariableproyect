# Interpretación geométrica de S(a, b) (§7.5–7.8)

Figuras en `results/images/` (se generan con `python src/main.py`).
Gráficas 3D: `12_3d_superficie_trazas.png`, `13_3d_curvas_nivel.png`, `14_3d_tazon_centrado.png`,
`15_3d_recta_y_superficie.png`, `16_3d_metodos_numericos.png`, `17_3d_contribucion_errores.png`.
Versión interactiva: `results/html/modelos_3d.html`.

## Idea clave: ¿qué forma tiene S(a, b)?

$$S(a,b)=63655\,a^2+1950\,ab+15\,b^2-268562\,a-4102\,b+283803$$

S es un **polinomio de grado 2** en (a, b) y su Hessiano es **constante y definido positivo**.
Por eso su gráfica z = S(a, b) es **siempre un paraboloide elíptico** (un "tazón"):

- tiene **un solo mínimo** (no hay mínimos locales "falsos");
- **no tiene ondulaciones**. Superficies onduladas como z = sen x · cos y tienen muchos
  máximos y mínimos, y su Hessiano cambia de punto a punto. La función de error de la
  regresión lineal nunca se ve así.

Esto responde la pregunta central: el problema de "encontrar la mejor recta" se convierte en
"encontrar el fondo de un tazón", y el Cálculo Multivariable lo resuelve exactamente con ∇S = 0
(el fondo es el único punto con plano tangente horizontal) y con el Hessiano (que garantiza que es
un fondo y no una silla ni un pico).

## 1. Superficie de error (§7.5–7.6) — `03_superficie_error.png`

- **Izquierda, variables originales (a, b):** se dibuja la región donde S ≤ 7·S_mín. No se ve como
  un tazón redondo sino como un **valle largo, estrecho y diagonal**. El mínimo (estrella roja)
  está en (3.45, −87.52) con S = 30.23 lb².
- **Intervalo adecuado:** la curva de nivel S = S_mín + Δ es la elipse ½·pᵀHp = Δ, que llega hasta
  ±√(2Δ·(H⁻¹)ₐₐ) ≈ ±0.80 en a, pero hasta ±√(2Δ·(H⁻¹)_bb) ≈ ±52 en b (con Δ = 6·S_mín). Por eso el rango en b
  es unas 65 veces más ancho que en a.
- **¿Por qué un valle?** Si aumentamos un poco la pendiente a, la recta se puede "compensar" bajando
  el intercepto b (la recta gira alrededor del centro de los datos, x̄ = 65). Muchas combinaciones
  (a, b) a lo largo de la línea b ≈ ȳ − a·x̄ dan errores parecidos: a y b están **acoplados**.
- **Derecha, x centrada:** si escribimos el modelo como y = a(x − x̄) + c, con c = b + a·x̄, el
  término cruzado desaparece (Σ(xᵢ − x̄) = 0), el Hessiano queda diagonal, H = diag(560, 30), y la
  misma función se ve como un **tazón casi redondo**. El problema es el mismo y el mínimo también;
  solo cambiamos los ejes.

## 2. Trazas y secciones transversales (§7.7) — `04_trazas.png`

- **Trazas con b = constante:** S(a, b₀) = 63655a² + (1950b₀ − 268562)a + … es una **parábola en a**
  que abre hacia arriba, con curvatura ∂²S/∂a² = 127310 (muy empinada). Con b₀ = b* el vértice está
  en a = 3.45. Con b₀ = b* ± 2 el vértice se corre a 3.419 o 3.481: **el mejor a depende de b**
  (consecuencia del término cruzado 1950ab).
- **Trazas con a = constante:** S(a₀, b) también es una parábola, pero mucho más plana
  (∂²S/∂b² = 30). Mover b en ±8 lb cambia S mucho menos que mover a en ±0.12.
- **Secciones por los ejes principales:** cortando la superficie por (a*, b*) en la dirección de cada
  autovector de H se obtienen dos parábolas idénticas en forma, pero a escalas muy distintas.
  En la dirección **empinada** (v₂ ≈ (−1, −0.015), casi solo a) S sube a 4·S_mín con apenas
  0.038 de desplazamiento. En la dirección **plana** (v₁ ≈ (0.015, −1), casi solo b, con a
  ajustándose) se necesitan ~37 unidades. La razón es ≈ √κ ≈ 980.

**Qué nos dicen:** todas las trazas son parábolas con vértice abajo, lo que confirma que S tiene un
mínimo y no un punto de silla (en una silla, una traza abriría hacia arriba y otra hacia abajo).

## 3. Curvas de nivel (§7.8) — `05_curvas_nivel.png`

- Las curvas S(a, b) = k son **elipses concéntricas** con centro común en (a*, b*). El punto
  calculado a mano está exactamente en el centro: es la región de mínimo error.
- **Distribución:** las elipses se separan cada vez más lentamente cuando k crece, porque S crece
  de forma cuadrática al alejarse del centro.
- **Forma:** en el plano (a, b) son extremadamente alargadas e inclinadas. La relación de ejes es
  √κ ≈ 980 y la inclinación sigue la dirección v₁. Con x centrada (panel derecho) las mismas
  curvas son elipses alineadas con los ejes, casi círculos.
- El gradiente ∇S es perpendicular a las curvas de nivel y apunta hacia afuera. En (a*, b*) vale
  cero: es el único punto donde no hay dirección de descenso.

## 4. ¿Por qué importa la forma? Métodos numéricos — `08_metodos_numericos.png`, `09_convergencia.png`

El descenso por gradiente "baja por la pendiente" (−∇S). En este valle tan alargado:

| Método | Iteraciones | Resultado |
|---|---|---|
| Gradiente simple en (a, b), desde (0, 0) | 200 000 (se detuvo) | (2.36, −16.35), **no converge** |
| Gradiente con x centrada | 550 | (3.4500, −87.5167) ✓ |
| Newton: (a,b) − H⁻¹∇S | **1** | (3.4500, −87.5167) exacto ✓ |

El gradiente simple cae rápido al fondo del valle y luego avanza casi nada a lo largo de él. Para
que no diverja, el paso η debe ser < 2/λ₂, y con ese paso avanzar en la dirección de λ₁ tomaría
del orden de κ ≈ 10⁶ iteraciones. Centrar x elimina el acoplamiento (κ baja a ≈ 19). Newton usa la
curvatura (el Hessiano) y, como S es cuadrática, su aproximación de Taylor de orden 2 es exacta:
llega en un paso.

**Conclusión geométrica:** el Hessiano no solo dice *si* hay un mínimo (signo de los autovalores),
también dice *cómo es* la superficie alrededor (las magnitudes de los autovalores), y eso
determina qué tan fácil es encontrar el mínimo numéricamente. La solución analítica (∇S = 0)
no tiene este problema: da el mínimo exacto sin importar la forma del valle.
