# Verificación computacional y análisis de resultados (§8–13)

Todo se reproduce con `python src/main.py --sin-gui` (la salida de consola trae cada tabla).

## §8 Qué hace el programa

| Paso pedido | Dónde está |
|---|---|
| 1. Cargar los datos | `src/datos.py` → `data/input/datos.csv` |
| 2. Calcular a*, b* con el método deducido (código propio) | `src/analitico.py` (`resolver_analitico`: fórmulas de las ecuaciones normales) |
| 3. Calcular S(a*, b*) | `src/modelo.py` (`error_cuadratico`) |
| 4. Datos + recta óptima + otras rectas | `02_regresion.png` |
| 5. Superficie, curvas de nivel, trazas, mínimo | `03`, `04`, `05` |
| 6. Calcular el Hessiano | `src/derivadas.py` (`hessiana`) |
| 7. Verificar numéricamente sus propiedades | `src/analisis.py` (`verificar_hessiano_numerico`) |
| Comparar con una biblioteca | `numpy.polyfit` y `scipy.stats.linregress` |

**Verificación numérica del Hessiano:** H calculado por diferencias finitas centradas coincide con la
fórmula (error relativo ≈ 1.6·10⁻¹²). Es simétrico, sus autovalores son λ₁ = 0.13193 y
λ₂ = 127 339.87 (ambos positivos ⇒ definido positivo ⇒ mínimo), det H = 16 800, y evaluado en
(0, 0) da la misma matriz, es decir, es constante.

## §9 Comparación analítico vs computacional

| Parámetro | Analítico (a mano) | Código propio | Biblioteca (scipy / numpy) | Diferencia máxima |
|---|---|---|---|---|
| a | 3.4500 | 3.4500000000 | 3.4500000000 | 0 |
| b | −87.5167 | −87.5166666667 | −87.5166666667 | 3.3·10⁻⁵ |
| S_mín | 30.2333 | 30.2333333333 | 30.2333333333 | 3.3·10⁻⁵ |

- La diferencia entre el cálculo manual y el código (3·10⁻⁵) se debe **solo al redondeo a 4
  decimales** hecho a mano. b* = −1312.75/15 es un decimal periódico.
- Entre el código propio y la biblioteca la diferencia es ≈ 10⁻¹⁴, es decir, error de punto
  flotante: **los tres métodos encuentran el mismo mínimo**.
- Los tres métodos coinciden porque resuelven el mismo sistema ∇S = 0. La biblioteca no usa
  "otro criterio": también minimiza la suma de cuadrados.

## §10 Análisis de la recta de ajuste — `02_regresion.png`

- La recta ŷ = 3.45x − 87.52 sigue muy bien la tendencia general: todos los puntos están a menos
  de 3.2 lb de ella, en un rango de pesos de 49 lb.
- Las otras rectas dibujadas (pendientes 3 o 4, recta desplazada 3 lb, recta por los extremos)
  tienen S entre 87 y 165 lb², de 3 a 5.5 veces más que la óptima (30.23). Ninguna recta tiene un
  error menor, como predice el Hessiano.
- **Mayores desviaciones:** x = 72 in (e = +3.12 lb), x = 58 in (e = +2.42) y x = 65 in (e = −1.73).
- **Puntos que influyen en el ajuste:** los extremos (58 y 72 in), porque están lejos de x̄ y
  "hacen palanca" sobre la pendiente (ver §13-B).

## §11 Análisis de residuos — `06_residuos_vs_x.png`, `07_histograma_residuos.png`

| eᵢ | x = 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 | 72 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lb | +2.42 | +0.97 | +0.52 | +0.07 | −0.38 | −0.83 | −1.28 | −1.73 | −1.18 | −1.63 | −1.08 | −0.53 | +0.02 | +1.57 | +3.12 |

- **¿Centrados en cero?** Sí, exactamente: Σeᵢ = 0. No es casualidad: es la ecuación ∂S/∂b = 0.
  Igual, Σxᵢeᵢ = 0 es ∂S/∂a = 0.
- **¿Tendencia o patrón?** **Sí, un patrón curvo en forma de U:** positivos en los extremos
  (residuo medio +0.72 lb en x ≤ 62 y +0.62 lb en x ≥ 68) y negativos en el centro (−1.33 lb).
  Solo hay 3 rachas de signo (+ + + + / − − − − − − − − / + + +), y corr(eᵢ, (xᵢ − x̄)²) = 0.97.
- **¿Cambia la dispersión con x?** Un poco: |e| medio de 0.92 lb en la primera mitad y 1.36 lb en
  la segunda, por el residuo grande en 72 in. No es un cambio sistemático de dispersión: es el
  efecto de la curvatura.
- **¿Valores atípicos?** x = 72 in tiene e/s = 2.04 (s = 1.525 lb), apenas por encima del umbral
  de 2. No es un error de medición: es el punto donde la curvatura se nota más.
- **¿Desviación sistemática?** **Sí.** La recta sobreestima el peso en el centro del rango
  (62–69 in) y lo subestima en los extremos. Los datos crecen un poco más rápido que una recta:
  el peso aumenta más por pulgada en las mujeres más altas.
- **Histograma:** centrado en 0 pero **asimétrico**: muchos residuos negativos pequeños (centro) y
  pocos positivos grandes (extremos). No tiene la forma de campana simétrica que se esperaría si
  los errores fueran solo ruido aleatorio.

**Conclusión de §11:** los residuos **no** son compatibles con un modelo lineal perfecto. Hay
estructura (curvatura) que la recta no captura. Aun así, es pequeña frente al rango de los datos.

## §12 Calidad del modelo

| Medida | Valor | Qué significa para estos datos |
|---|---|---|
| RMSE | **1.42 lb** (≈ 0.64 kg) | Error típico de la recta al predecir el peso promedio. Es el 1.0 % del peso medio (136.7 lb) y el 2.9 % del rango (49 lb). |
| R² | **0.9910** | La recta explica el 99.1 % de la variación del peso promedio entre estaturas. |
| MAE | 1.16 lb | Error absoluto medio. |

**Cómo interpretarlo (no basta decir "R² es bueno"):**
- El R² es muy alto porque los datos son **promedios** de muchas personas. El ruido individual ya
  se canceló y la variable x abarca un rango amplio. Con datos de personas individuales el R² sería
  mucho menor: dos mujeres de la misma estatura pueden diferir en 20 lb o más.
- Un R² de 0.99 **no prueba que el modelo sea correcto**: el análisis de residuos muestra una
  curvatura sistemática. R² mide cuánta variación se explica, no si la forma del modelo es la
  adecuada.
- **Según el propósito:** para estimar el peso promedio a partir de la estatura dentro de 58–72 in
  la recta es suficiente, porque un error de ±1.4 lb es menor que la variación normal de peso de
  una persona en un mismo día. Para **extrapolar** fuera del rango (por ejemplo 80 in o 50 in) no
  es confiable, y b* = −87.5 lb muestra que la recta no tiene sentido cerca de x = 0.

## §13 Análisis adicional (se hicieron las opciones A, B y C)

### Opción B — Influencia de un dato — `10_influencia.png`

Se eliminó cada punto, uno a la vez, y se recalculó la recta:

| Punto eliminado | a sin el punto | Δa | b sin el punto | Δb |
|---|---|---|---|---|
| x = 58 (extremo) | 3.5297 | +0.080 | −92.91 | −5.39 |
| x = 65 (centro) | 3.4500 | 0.000 | −87.39 | +0.12 |
| x = 72 (extremo) | 3.3473 | **−0.103** | −81.11 | **+6.41** |

- El punto **más influyente es x = 72 in**: sin él la pendiente baja 3 %, y la predicción cambia
  hasta 1 lb.
- Los puntos centrales casi no influyen en la pendiente. Quitar x = 65 = x̄ no la cambia en nada.
  La pendiente es un promedio ponderado por (xᵢ − x̄), y por eso los extremos pesan más.
- Ningún punto cambia la recta de forma drástica (|Δa| ≤ 3 %): el ajuste es **robusto**.

### Opción C — Comparación con un modelo cuadrático — `11_lineal_vs_cuadratico.png`

Se minimizó S(p, q, r) = Σ(yᵢ − pxᵢ² − qxᵢ − r)² con el mismo método: ∇S = 0 da un sistema 3×3,
resuelto con código propio y verificado con `polyfit`.

| | Lineal | Cuadrático |
|---|---|---|
| Modelo | ŷ = 3.45x − 87.52 | ŷ = 0.0831x² − 7.348x + 261.88 |
| S | 30.23 lb² | **1.77 lb²** |
| R² | 0.99101 | 0.99947 |
| RMSE | 1.42 lb | 0.34 lb |

- El modelo cuadrático reduce el error **17 veces**, y sus residuos ya no muestran el patrón en U.
  La prueba F parcial (F = 193, p ≈ 10⁻⁸) indica que la mejora no es casualidad: **la curvatura
  es real**.
- Aun así, las predicciones de ambos modelos difieren **como máximo 2.5 lb** dentro del rango. Para
  uso práctico (estimar el peso promedio) la recta es suficiente y mucho más fácil de interpretar
  (3.45 lb por pulgada). Para describir la forma exacta de la relación, el cuadrático es mejor.

### Opción A — Sensibilidad

Se sumó +1 lb a cada yᵢ, uno a la vez:
- |Δa| máximo = 0.025 lb/in (0.7 %), al modificar un extremo.
- |Δb| máximo = 1.69 lb.

Como a* = Σ wᵢyᵢ con wᵢ = (xᵢ − x̄)/Σ(xᵢ − x̄)², el cambio es lineal y proporcional a la distancia
a x̄. La pendiente es **poco sensible** a errores pequeños en los datos. El intercepto es más
sensible porque está lejos de los datos (en x = 0).
