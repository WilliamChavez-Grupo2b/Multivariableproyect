# Guía de la app interactiva `app.html`

Explica **todo** lo que muestra `results/html/app.html`: los datos y las variables, las ecuaciones, los métodos
de cálculo, cada una de las nueve vistas y, con detalle, las **cuatro gráficas 3D** de la vista 4.

Todos los números de ejemplo corresponden al conjunto real del proyecto (estatura vs. peso, n = 15) y fueron
verificados con el código (`src/` y `web/regresion.js`).

---

## Índice

1. [Qué es la app y cómo abrirla](#1-qué-es-la-app-y-cómo-abrirla)
2. [Los datos](#2-los-datos)
3. [Diccionario de variables](#3-diccionario-de-variables)
4. [Ecuaciones del modelo](#4-ecuaciones-del-modelo)
5. [Métodos de cálculo](#5-métodos-de-cálculo)
6. [Recorrido por las nueve vistas](#6-recorrido-por-las-nueve-vistas)
7. [Las cuatro gráficas 3D y su significado](#7-las-cuatro-gráficas-3d-y-su-significado)
8. [Cómo leer los resultados del caso real](#8-cómo-leer-los-resultados-del-caso-real)
9. [Guion sugerido para la sustentación](#9-guion-sugerido-para-la-sustentación)

---

## 1. Qué es la app y cómo abrirla

`app.html` es **un solo archivo** (≈ 4.9 MB) que funciona **sin internet**: se descarga y se abre con doble clic
en cualquier navegador. Lleva dentro tres piezas:

| Pieza | Qué hace |
|---|---|
| **Datos de Python** (JSON incrustado) | Resultados oficiales calculados por `src/` : a*, b*, S*, Hessiano, bibliotecas (`numpy.polyfit`, `numpy.linalg.lstsq`, `scipy.stats.linregress`), modelo cuadrático y su p-valor. |
| **`regresion.js`** | Las mismas fórmulas, escritas en JavaScript, para recalcular al instante cuando se mueven los controles o se editan los datos. |
| **Plotly.js** | Biblioteca de gráficos 2D y 3D interactivos (girar, acercar, descargar PNG). |

**Insignia de paridad.** Al abrir, la app recalcula a*, b* y S* en el navegador y los compara con los de Python.
Si coinciden (diferencia ≤ 10⁻⁹) aparece en verde **"Paridad Python ↔ navegador ✓"**. Esto demuestra que lo que
se ve en pantalla es exactamente lo que calculó el programa. La misma comprobación es la prueba automática 13.

**Regenerarla:** `python src/app_html.py` (o `python src/main.py`, que genera todo).

---

## 2. Los datos

| Aspecto | Valor |
|---|---|
| Problema | Relación entre la **estatura** y el **peso promedio** de mujeres estadounidenses de 30 a 39 años |
| Fuente | *The World Almanac and Book of Facts* (1975); conjunto `women` de R |
| Tamaño | n = 15 observaciones |
| Variable independiente | x = estatura, en pulgadas (in) — 1 in = 2.54 cm |
| Variable dependiente | y = peso promedio, en libras (lb) — 1 lb = 0.4536 kg |

| i | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xᵢ [in] | 58 | 59 | 60 | 61 | 62 | 63 | 64 | 65 | 66 | 67 | 68 | 69 | 70 | 71 | 72 |
| yᵢ [lb] | 115 | 117 | 120 | 123 | 126 | 129 | 132 | 135 | 139 | 142 | 146 | 150 | 154 | 159 | 164 |

**Dos características importantes de estos datos:**

1. Cada yᵢ es un **promedio** de muchas personas, no una persona. Por eso los puntos están tan alineados
   (los promedios eliminan la variación individual) y R² sale muy alto.
2. Los incrementos de y crecen con x (3, 3, 3 … y luego 4, 4, 5, 5): la relación es **ligeramente curva**.
   La app lo detecta en los residuos (vista 7) y en el modelo cuadrático (vista 9-C).

---

## 3. Diccionario de variables

### 3.1 Datos y modelo

| Símbolo | Nombre | Unidad | Significado |
|---|---|---|---|
| n | Número de observaciones | — | Cantidad de pares (xᵢ, yᵢ) **activos**. Debe ser ≥ 3. |
| xᵢ, yᵢ | Dato i | in, lb | Estatura y peso de la observación i. |
| a | Pendiente | lb/in | Cuánto aumenta el peso por cada pulgada de estatura. |
| b | Intercepto | lb | Valor de la recta cuando x = 0 (sin sentido físico aquí: nadie mide 0 in; es un parámetro de ajuste). |
| ŷᵢ | Valor predicho | lb | ŷᵢ = a·xᵢ + b. |
| eᵢ | Residuo | lb | eᵢ = yᵢ − ŷᵢ. Positivo: el dato está **sobre** la recta; negativo: **bajo** la recta. |
| S(a, b) | Error cuadrático total | lb² | Suma de los residuos al cuadrado. Es la **función de dos variables** que se minimiza. |
| a*, b* | Parámetros óptimos | lb/in, lb | El punto donde S es mínima: a* = 3.45, b* = −87.5167. |
| S* | Error mínimo | lb² | S(a*, b*) = 30.2333. |

### 3.2 Sumatorias (vista 1 y 2)

| Símbolo | Fórmula | Valor |
|---|---|---|
| Σx | Σ xᵢ | 975 |
| Σy | Σ yᵢ | 2051 |
| Σxy | Σ xᵢ·yᵢ | 134 281 |
| Σx² | Σ xᵢ² | 63 655 |
| Σy² | Σ yᵢ² | 283 803 |
| x̄ | Σx / n | 65 |
| ȳ | Σy / n | 136.7333 |
| D | n·Σx² − (Σx)² | 4200 |

### 3.3 Variables centradas (gráfica 3D nº 3)

| Símbolo | Fórmula | Significado |
|---|---|---|
| uᵢ | xᵢ − x̄ | Estatura medida **desde el promedio** (−7 … +7). |
| c | b + a·x̄ | Intercepto en las variables centradas: es la altura de la recta **en x = x̄**. En el óptimo c* = ȳ = 136.7333. |
| Σu² | Σx² − n·x̄² | 280 |
| Σuy | Σxy − x̄·Σy | 966 |

### 3.4 Hessiano y geometría

| Símbolo | Significado | Valor |
|---|---|---|
| H | Matriz de segundas derivadas de S | [[127 310, 1950], [1950, 30]] |
| det H | Determinante | 16 800 |
| λ₁, λ₂ | Valores propios de H (curvaturas principales) | 0.1319 y 127 339.87 |
| v₁, v₂ | Vectores propios (direcciones de curvatura mínima y máxima) | v₁ ≈ (0.0153, −0.9999), v₂ ≈ (0.9999, 0.0153) |
| κ | Número de condición λ₂ / λ₁ | ≈ 965 205 |

### 3.5 Calidad y residuos

| Símbolo | Fórmula | Valor | Significado |
|---|---|---|---|
| SS_res | Σ eᵢ² | 30.2333 lb² | Variación que la recta **no** explica (= S*). |
| SS_tot | Σ (yᵢ − ȳ)² | 3362.9333 lb² | Variación total de y alrededor de su media. |
| R² | 1 − SS_res / SS_tot | 0.99101 | Fracción de la variación de y explicada por la recta (99.10 %). |
| RMSE | √(SS_res / n) | 1.4197 lb | Error típico de predicción, en libras. |
| MAE | Σ\|eᵢ\| / n | 1.1556 lb | Error absoluto promedio. |
| s | √(SS_res / (n − 2)) | 1.5250 lb | Error estándar de la regresión (divide entre n − 2 porque se estimaron 2 parámetros). |
| zᵢ | eᵢ / s | — | Residuo estandarizado; \|zᵢ\| > 2 se marca como atípico. |

---

## 4. Ecuaciones del modelo

### 4.1 Modelo y función de error

$$\hat y_i = a\,x_i + b \qquad e_i = y_i - (a\,x_i + b)$$

$$S(a,b) = \sum_{i=1}^{n} \left(y_i - a\,x_i - b\right)^2$$

Se elevan al cuadrado para que los errores positivos y negativos no se cancelen y para que S sea derivable.

### 4.2 Derivadas parciales (regla de la cadena)

$$\frac{\partial S}{\partial a} = \sum 2\,(y_i - a x_i - b)\,(-x_i) = -2\sum x_i\,(y_i - a x_i - b) = 2\left(a\,\Sigma x^2 + b\,\Sigma x - \Sigma xy\right)$$

$$\frac{\partial S}{\partial b} = \sum 2\,(y_i - a x_i - b)\,(-1) = -2\sum (y_i - a x_i - b) = 2\left(a\,\Sigma x + n\,b - \Sigma y\right)$$

### 4.3 Punto crítico: ∇S = 0 (ecuaciones normales)

$$\begin{cases} a\,\Sigma x^2 + b\,\Sigma x = \Sigma xy \\ a\,\Sigma x + n\,b = \Sigma y \end{cases}
\quad\Longrightarrow\quad
\begin{cases} 63655\,a + 975\,b = 134281 \\ 975\,a + 15\,b = 2051 \end{cases}$$

**Consecuencias de ∇S = 0** (se ven en la vista 7): Σeᵢ = 0 (los residuos suman cero) y Σxᵢeᵢ = 0.

### 4.4 Solución

$$a^* = \frac{n\,\Sigma xy - \Sigma x\,\Sigma y}{n\,\Sigma x^2 - (\Sigma x)^2} = \frac{15(134281) - 975(2051)}{15(63655) - 975^2} = \frac{14490}{4200} = 3.45$$

$$b^* = \bar y - a^*\bar x = 136.7333 - 3.45(65) = -87.5167$$

$$\boxed{\hat y = 3.45\,x - 87.5167} \qquad S^* = 30.2333\ \text{lb}^2$$

**Interpretación:** por cada pulgada adicional de estatura, el peso promedio aumenta 3.45 lb.

### 4.5 Hessiano y criterio del mínimo

$$H = \begin{pmatrix} \dfrac{\partial^2 S}{\partial a^2} & \dfrac{\partial^2 S}{\partial a\,\partial b} \\[2mm] \dfrac{\partial^2 S}{\partial b\,\partial a} & \dfrac{\partial^2 S}{\partial b^2}\end{pmatrix}
= 2\begin{pmatrix} \Sigma x^2 & \Sigma x \\ \Sigma x & n \end{pmatrix}
= \begin{pmatrix} 127310 & 1950 \\ 1950 & 30 \end{pmatrix}$$

$$\det H = 4\left(n\,\Sigma x^2 - (\Sigma x)^2\right) = 4(4200) = 16800$$

**Criterio:** ∂²S/∂a² = 127 310 > 0 y det H = 16 800 > 0 ⇒ H es **definida positiva** ⇒ (a*, b*) es un **mínimo**.
Como H **no depende de a ni de b** (S es un polinomio de grado 2), es definida positiva en todo el plano:
S es convexa y el mínimo es **global y único**.

**¿Cuándo falla?** det H = 0 solo si todos los xᵢ son iguales; entonces no existe una recta única. La app lo
detecta y muestra el error.

### 4.6 S escrita con sumatorias (la usa la superficie 3D)

Desarrollando el cuadrado:

$$S(a,b) = \Sigma y^2 - 2a\,\Sigma xy - 2b\,\Sigma y + a^2\,\Sigma x^2 + 2ab\,\Sigma x + n\,b^2$$

Con esta forma cada punto de la malla 3D cuesta 6 operaciones, sin recorrer los datos.
Muestra también que S es un **polinomio cuadrático** en (a, b): su gráfica es un **paraboloide elíptico**.

### 4.7 Forma centrada

Con uᵢ = xᵢ − x̄ y c = b + a·x̄, como Σu = 0:

$$S(a,c) = \Sigma y^2 - 2a\,\Sigma uy - 2c\,\Sigma y + a^2\,\Sigma u^2 + n\,c^2 \qquad
H_c = \begin{pmatrix} 2\Sigma u^2 & 0 \\ 0 & 2n\end{pmatrix} = \begin{pmatrix} 560 & 0 \\ 0 & 30 \end{pmatrix}$$

El Hessiano queda **diagonal**: a y c se optimizan por separado,
a* = Σuy / Σu² = 966 / 280 = 3.45 y c* = ȳ = 136.7333, y κ baja de 965 205 a **18.7**.

### 4.8 Calidad del ajuste

$$R^2 = 1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2} \qquad
\text{RMSE} = \sqrt{\frac{1}{n}\sum e_i^2} \qquad
\text{MAE} = \frac{1}{n}\sum |e_i|$$

Si todos los yᵢ son iguales, SS_tot = 0 y R² **no está definido**: la app muestra "—".

### 4.9 Modelo cuadrático (vista 9-C)

Para y = p·x² + q·x + r se minimiza S(p, q, r) = Σ(yᵢ − p xᵢ² − q xᵢ − r)². ∇S = 0 da un sistema 3×3:

$$\begin{pmatrix} \Sigma x^4 & \Sigma x^3 & \Sigma x^2 \\ \Sigma x^3 & \Sigma x^2 & \Sigma x \\ \Sigma x^2 & \Sigma x & n \end{pmatrix}
\begin{pmatrix} p \\ q \\ r \end{pmatrix} = \begin{pmatrix} \Sigma x^2 y \\ \Sigma xy \\ \Sigma y \end{pmatrix}$$

**Prueba F parcial** (¿el término x² mejora el ajuste de forma significativa?):

$$F = \frac{S_{\text{lineal}} - S_{\text{cuad}}}{S_{\text{cuad}} / (n - 3)}$$

---

## 5. Métodos de cálculo

| Método | Dónde se usa | Cómo funciona |
|---|---|---|
| **Ecuaciones normales (fórmula cerrada)** | Vistas 1–9 (código propio) | Fórmulas de §4.4 con las sumatorias. Es el método **oficial**; no usa funciones de regresión de biblioteca. |
| **Sistema lineal 2×2** | Python (`resolver_sistema_matricial`) | Resuelve §4.3 con `numpy.linalg.solve` como verificación. |
| **Bibliotecas** | Vista 8 | `numpy.polyfit(x, y, 1)`, `numpy.linalg.lstsq([x 1], y)` y `scipy.stats.linregress`. Solo verifican. |
| **Diferencias finitas centradas** | Vista 6 | Aproxima las segundas derivadas con h = 10⁻³: ∂²S/∂a² ≈ [S(a+h,b) − 2S(a,b) + S(a−h,b)] / h². Se evalúa en (a*, b*) y en (0, 0) para mostrar que H es constante. |
| **Valores y vectores propios 2×2** | Vista 6 | λ = traza/2 ± √((H₁₁−H₂₂)²/4 + H₁₂²); λ₁ se calcula como det/λ₂ para evitar errores de redondeo. |
| **Descenso por gradiente (x centrada)** | Vista 3 y gráfica 3D nº 4 | (a, c) ← (a, c) − η·∇S, con η = 1 / traza(H_c) = 1/590. Se detiene cuando ‖∇S‖ < 10⁻⁶. |
| **Eliminación de Gauss con pivoteo** | Vista 9-C | Resuelve el sistema 3×3 del modelo cuadrático. |
| **Leave-one-out** | Vista 9-B | Se quita un punto, se reajusta y se mide el cambio. |
| **Perturbación** | Vista 9-A | Se suma δ a un yᵢ (uno a la vez) y se reajusta. |
| **Submuestreo aleatorio con semilla** | Vista 9-D | Para cada m = 3…n se toman 200 subconjuntos aleatorios (semilla fija → reproducible). |
| **Regla de Sturges** | Vista 7 | Número de clases del histograma: k = ⌈1 + log₂ n⌉ = 5. |

**Rango de la malla 3D.** La superficie se dibuja alrededor del óptimo, siguiendo el valle:

$$\Delta a = \text{zoom}\cdot\max\!\left(10\,\frac{s}{\sqrt{\Sigma u^2}},\ 0.25\,|a^*|\right) \approx 0.911 \qquad
\Delta b = \Delta a\left(|\bar x| + \sqrt{\Sigma u^2/n}\right) \approx 63.2 \qquad
\Delta c = \Delta a\sqrt{\Sigma u^2/n} \approx 3.94$$

Así a ∈ [2.54, 4.36], b ∈ [−150.7, −24.3] y, en variables centradas, c ∈ [132.8, 140.7]. Se usa una malla de 60 × 60 puntos.

---

## 6. Recorrido por las nueve vistas

### Controles (a, b) compartidos — vistas 3, 4 y 5

Dos deslizadores (más campos para escribir el valor exacto) fijan una recta cualquiera **y = a·x + b**. Las tres vistas
se actualizan juntas: es el **mismo punto (a, b)** visto como recta, como punto en el plano de parámetros y como punto
sobre la superficie.

* **Indicador:** S(a, b), S* y cuánto por encima del mínimo se está (en %).
* **Barra:** se llena al 100 % en el mínimo y baja en escala logarítmica (100 − 25·log₁₀(S/S*)).
* **"Ir al óptimo"** lleva a (a*, b*). **"Recta de prueba"** vuelve a la recta inicial a = a* + 0.35·Δa ≈ 3.769, b = b*,
  que tiene S ≈ 6506.9 (21 422 % por encima del mínimo): la misma b pero algo más de pendiente ya da un error enorme.

### Vista 1 · Datos (§5)

* **Tabla editable:** cambiar un valor, desmarcar "activo" para excluir un punto, borrar o agregar filas.
* **Cargar CSV:** acepta columnas `x,y` (con coma o punto y coma). **Descargar CSV** guarda los datos activos.
* **Diagrama de dispersión** con la recta óptima; los puntos excluidos aparecen huecos.
* **Tarjetas:** n, Σx, Σy, Σxy, Σx², Σy², x̄, ȳ.
* **Validación:** n ≥ 3, valores numéricos y x no todos iguales; si falla aparece un aviso rojo y las demás vistas se pausan.

### Vista 2 · Analítico (§7.1–7.4)

Reproduce el procedimiento manual **con los números de los datos actuales**: S(a, b), las dos derivadas parciales, el
sistema ∇S = 0, la solución a*, b*, S*, el valor de ∇S(a*, b*) (≈ 10⁻¹¹, prueba de que es punto crítico) y el criterio
del Hessiano. A la derecha se escriben los valores **a mano** (a_man, b_man, S_man) y se muestra su diferencia con el código.

### Vista 3 · Explorador

* **Izquierda:** los datos, la recta óptima (roja), la recta (a, b) elegida (azul) y sus **residuos como segmentos
  verticales punteados**. S es la suma de los cuadrados de esas longitudes.
* **Derecha:** las **curvas de nivel** de log₁₀ S en el plano (a, b), la estrella en (a*, b*) y el punto azul (a, b).
  **Haciendo clic** en el mapa se mueve el punto. En la leyenda se puede activar la trayectoria del descenso por gradiente.
* **Idea clave:** acercar la recta a los datos es lo mismo que acercar el punto al centro de las elipses.

### Vista 4 · Superficie 3D (§7.5–7.8)

Ver la [sección 7](#7-las-cuatro-gráficas-3d-y-su-significado).

### Vista 5 · Trazas y secciones transversales (§7.7)

Son cortes de la superficie con planos verticales:

| Gráfica | Corte | Ecuación | Vértice | Curvatura |
|---|---|---|---|---|
| Izquierda | b = b₀ fijo | S(a; b₀) parábola en a | a_v = (Σxy − b₀Σx) / Σx² | ∂²S/∂a² = 2Σx² = 127 310 (muy cerrada) |
| Derecha | a = a₀ fijo | S(a₀; b) parábola en b | b_v = (Σy − a₀Σx) / n | ∂²S/∂b² = 2n = 30 (muy abierta) |

La línea punteada roja es la traza que pasa **por el mínimo** (b₀ = b* o a₀ = a*); solo en ese caso el vértice coincide
con a* o b*. **Lección:** fijar una variable y minimizar en la otra no da el óptimo, salvo que la variable fija ya esté en
su valor óptimo. Las dos curvaturas tan distintas (127 310 vs 30) explican por qué la superficie es un valle estrecho.

### Vista 6 · Hessiano (§7.4, §8.6–8.7)

* Tabla con cada segunda derivada: **fórmula, valor analítico, diferencias finitas en (a*, b*) y en (0, 0)**. Coinciden
  con error relativo ≈ 10⁻¹² y son iguales en ambos puntos: **H es constante**.
* Tarjetas: det H, traza, λ₁, λ₂ y κ.
* Criterio con ✓: ∂²S/∂a² > 0, det H > 0, λ₁, λ₂ > 0 y simetría ⇒ mínimo global.
* **Gráfica:** los vectores propios dibujados sobre las curvas de nivel. v₁ (λ₁ = 0.13) apunta **a lo largo del valle**
  (allí S casi no cambia); v₂ (λ₂ = 127 340) apunta a través del valle (allí S crece muy rápido). Como los ejes a y b
  tienen escalas muy distintas, en pantalla no se ven perpendiculares aunque lo son.

### Vista 7 · Residuos y calidad (§10–12)

* **Tarjetas:** R², RMSE, MAE, RMSE/ȳ, media de eᵢ, rachas de signo, corr(e, (x − x̄)²) y número de atípicos.
* **Residuos vs x:** triángulos hacia arriba (eᵢ ≥ 0) y hacia abajo (eᵢ < 0), bandas ± 2s y círculo rojo en los atípicos.
* **Histograma** con 5 clases (Sturges).
* **Interpretación automática** de los cinco puntos del §11: centrado, patrón, regiones, dispersión y atípicos.
* **Tabla de residuos** (x, y, ŷ, e, z) descargable en CSV.

| Indicador | Valor | Qué significa |
|---|---|---|
| Rachas de signo | 3 de 15 | Positivos al inicio, negativos en el centro, positivos al final: no es azar. |
| corr(e, (x − x̄)²) | 0.970 | Los residuos crecen con la distancia al centro: **forma de U**. |
| Residuo medio por regiones | +0.72 / −1.33 / +0.62 lb | Extremos sobre la recta, centro debajo: **desviación sistemática**. |
| \|e\| medio por mitades | 0.92 / 1.36 lb | La dispersión aumenta un poco con x. |
| Atípico | x = 72 (z = 2.04) | Justo en el límite; es el extremo de la curva, no un error de medición. |

### Vista 8 · Comparación (§9)

| Parámetro | A mano | Código (navegador) | Código (Python) | polyfit | lstsq | linregress | Δ máx |
|---|---|---|---|---|---|---|---|
| a | 3.45 | 3.45 | 3.45 | 3.45 | 3.45 | 3.45 | ≈ 2·10⁻¹⁵ |
| b | −87.5167 | −87.516667 | −87.516667 | −87.516667 | −87.516667 | −87.516667 | 3.3·10⁻⁵ |
| S_mín | 30.2333 | 30.233333 | 30.233333 | 30.233333 | 30.233333 | 30.233333 | 3.3·10⁻⁵ |

La diferencia con el cálculo a mano (≈ 3·10⁻⁵) es solo **redondeo a 4 decimales**; entre código y bibliotecas es error de
punto flotante (≈ 10⁻¹³). Si se editan los datos en la vista 1, las columnas de Python y de bibliotecas muestran "—"
porque corresponden al archivo original.

### Vista 9 · Análisis adicional (§13)

**A · Sensibilidad.** Con el control δ se suma δ libras a cada yᵢ, uno a la vez. Como a* = Σwᵢyᵢ con
wᵢ = (xᵢ − x̄)/Σu², el cambio es **lineal** y depende de la distancia al centro: con δ = 1 lb,
Δa* = ±0.025 en los extremos (x = 58 y 72) y 0 en el centro (x = 65). La pendiente es poco sensible a errores pequeños.

**B · Influencia.** Se elimina cada punto y se reajusta; la barra mide el mayor cambio de la predicción en el rango.
El más influyente es **x = 72** (máx |Δŷ| = 0.99 lb, Δa = −0.103, Δb = +6.41), seguido de x = 58 (0.77 lb).
Los extremos pesan más (como una palanca). **Al hacer clic en una barra** se excluye ese punto y toda la app se recalcula.

**C · Lineal vs cuadrático.**

| | Lineal | Cuadrático |
|---|---|---|
| Ecuación | ŷ = 3.45x − 87.5167 | ŷ = 0.08306x² − 7.3483x + 261.878 |
| S | 30.2333 | 1.7701 |
| R² | 0.99101 | 0.99947 |
| RMSE | 1.4197 lb | 0.3435 lb |

F = 192.96 con p ≈ 9.3·10⁻⁹: la curvatura es **estadísticamente real**. Pero las predicciones de ambos modelos difieren
a lo sumo 2.52 lb: para estimar un peso promedio la recta es suficiente; para describir la forma exacta, el cuadrático es mejor.

**D · Efecto del número de datos.** Con m datos elegidos al azar (200 repeticiones por m), la desviación de a* baja de
0.398 (m = 3) a 0.034 (m = 14) y a 0 con los 15 datos. Más datos ⇒ parámetros más estables. El promedio de a* se mantiene
cerca de 3.45 para todo m: el método no tiene sesgo, solo menos precisión con pocos datos.

---

## 7. Las cuatro gráficas 3D y su significado

La vista 4 tiene **una figura 3D con cuatro representaciones** que se activan con las casillas y la leyenda. Todas
muestran la misma idea: **cada recta posible es un punto (a, b) del plano, y su altura es el error S(a, b) que comete**.
La mejor recta es el punto más bajo.

**Controles comunes:** girar arrastrando, acercar con la rueda, botones de cámara (*Isométrica*, *Desde arriba*,
*Frente a a*, *Frente a b*), **"Eje z en log₁₀"**, **"Variables centradas"** y **"Zoom del rango"**. Al pasar el cursor
se lee a, b (o c) y el valor real de S. En la leyenda, un clic oculta o muestra cada capa.

### 7.1 Gráfica 3D nº 1 — Superficie de error z = S(a, b) con curvas de nivel

**Qué se dibuja:**
- La superficie z = S(a, b) sobre la malla 60 × 60, coloreada por altura (escala Viridis: morado = bajo, amarillo = alto).
- Las **curvas de nivel proyectadas en el piso** (cortes horizontales S = constante).
- El **rombo rojo** en (a*, b*, S*) y el **punto azul** en la recta elegida con los controles.

**Ecuación:** z = Σy² − 2aΣxy − 2bΣy + a²Σx² + 2abΣx + n b² (§4.6), un **paraboloide elíptico**.

**Qué significa:**
- La superficie tiene **un solo punto más bajo**: el mínimo global que garantiza el Hessiano.
- Las curvas de nivel son **elipses concéntricas** centradas en (a*, b*). Sus ejes son los vectores propios de H.
- Las elipses son **extremadamente alargadas** (κ ≈ 965 205): la superficie es un **valle largo y estrecho**, no un tazón
  redondo. Moverse a lo largo del valle casi no cambia S; atravesarlo la cambia muchísimo.

**Por qué el eje z en log₁₀:** S va de 30 en el mínimo a millones en los bordes. En escala lineal todo el fondo se
ve plano y el mínimo no se distingue; con log₁₀ S el fondo del valle se ve con detalle. Al desactivar la casilla se ve
la forma "real" de parábola (paredes muy empinadas).

### 7.2 Gráfica 3D nº 2 — Planos de corte y trazas

**Qué se dibuja** (aparece por defecto con el punto (a, b) de los controles):
- **Plano azul a = a₀** y **plano naranja b = b₀**, verticales y semitransparentes.
- Sus **intersecciones con la superficie** como curvas gruesas: la **traza a = a₀** (azul) y la **traza b = b₀** (naranja).

**Ecuaciones:**

$$S(a_0, b) = n\,b^2 - 2b\,(\Sigma y - a_0\Sigma x) + \text{cte} \qquad S(a, b_0) = \Sigma x^2\,a^2 - 2a\,(\Sigma xy - b_0\Sigma x) + \text{cte}$$

Ambas son **parábolas que abren hacia arriba** (coeficiente principal n > 0 y Σx² > 0).

**Qué significa:**
- Es la versión 3D de la vista 5: las trazas son "rebanadas" de la superficie.
- Al mover los controles se ve cómo los planos recorren la superficie. Solo cuando a₀ = a* y b₀ = b* las dos trazas
  tienen su vértice en el mismo punto: el mínimo.
- La traza naranja (en a) es una parábola muy cerrada; la azul (en b) es muy abierta. Esa diferencia de curvaturas
  (127 310 vs 30) es lo que hace el valle tan estrecho.

### 7.3 Gráfica 3D nº 3 — El "tazón" con x centrada

**Cómo se activa:** casilla **"Variables centradas (a, c), u = x − x̄"**.

**Qué se dibuja:** la misma función de error, pero con los parámetros (a, c), donde c = b + a·x̄ es la altura de la recta
en el punto medio x̄ = 65.

**Ecuación:** S(a, c) = Σy² − 2aΣuy − 2cΣy + a²Σu² + n c², con Hessiano **diagonal** H_c = diag(560, 30).

**Qué significa:**
- Es **la misma familia de rectas**, solo descrita con otros parámetros. El mínimo da la misma recta: a* = 3.45 y
  c* = ȳ = 136.73 (⇒ b* = c* − a*x̄ = −87.5167).
- Como el término cruzado desaparece, las elipses quedan **alineadas con los ejes** y κ baja a **18.7**: la superficie
  se ve como un **tazón** casi redondo en lugar de un valle estrecho.
- **Por qué el valle original es tan estrecho:** a y b están acoplados. Si la pendiente sube un poco, el intercepto debe
  bajar 65 veces más (x̄ = 65) para que la recta siga pasando por el centro de los datos. Centrar x elimina ese acoplamiento.
- Consecuencia práctica: los métodos numéricos funcionan mucho mejor con x centrada (siguiente gráfica).

*Nota:* la recta de prueba tiene c ≈ 157.5, fuera del rango del tazón; la app oculta su punto y planos y lo avisa.
Con **"Ir al óptimo"** el punto vuelve a verse.

### 7.4 Gráfica 3D nº 4 — Valle y trayectoria del descenso por gradiente

**Cómo se activa:** en la leyenda, clic en **"Valle b = ȳ − a·x̄"** (línea blanca discontinua) y en
**"Descenso por gradiente"** (línea verde con puntos).

**Qué se dibuja:**
- **El valle:** la curva de puntos (a, b) que minimizan S para cada a fijo, b = ȳ − a·x̄. Es el "fondo" del valle;
  todas las rectas de esta familia pasan por el punto medio (x̄, ȳ) = (65, 136.73).
- **La trayectoria del descenso por gradiente:** los puntos (a_k, b_k, S_k) que recorre el método numérico desde
  (2.812, −43.29), donde S = 258.1, hasta el mínimo.

**Método:**

$$(a, c)_{k+1} = (a, c)_k - \eta\,\nabla S(a, c)_k \qquad \eta = \frac{1}{\operatorname{traza}(H_c)} = \frac{1}{590}$$

La iteración se hace en variables centradas y se dibuja en las originales. Converge en **351 iteraciones**
(‖∇S‖ < 10⁻⁶) al mismo (a*, b*) = (3.45, −87.5167).

**Qué significa:**
- Se ve **cómo** se llega al mínimo, en dos etapas. La pendiente se corrige casi de inmediato: en 3 iteraciones
  a ≈ 3.45, porque en esa dirección la curvatura es grande (2Σu² = 560) y cada paso reduce el error de a en un 95 %.
  Después el método ajusta el intercepto despacio (b va de −82.8 a −87.52), porque en la dirección de c la curvatura es
  pequeña (2n = 30) y cada paso reduce el error solo un 5 %. Por eso la trayectoria primero se desplaza casi solo en
  la dirección de a y luego casi solo en la de b, hasta el mínimo, mientras S pasa de 258 a 30.23.
- El gradiente siempre apunta en la dirección de máximo crecimiento; avanzar en sentido contrario baja por la superficie.
  En el mínimo ∇S = 0 y el método se detiene: es la misma condición que se resolvió a mano.
- **Por qué con x centrada:** en las variables originales κ ≈ 965 205 y el descenso simple **no converge** ni en 200 000
  iteraciones (se queda en a ≈ 2.36, b ≈ −16.4, avanzando muy despacio por el valle). Con x centrada, κ = 18.7 y basta con
  351. Es una consecuencia directa de la forma geométrica que describe el Hessiano.
- La solución analítica sigue siendo la oficial; el método numérico confirma que el punto crítico es alcanzable "bajando".

### Resumen de las cuatro gráficas

| Nº | Representación | Pregunta que responde | Concepto del curso |
|---|---|---|---|
| 1 | Superficie + curvas de nivel | ¿Dónde está el menor error? | Función de dos variables, mínimo, curvas de nivel (§7.5–7.8) |
| 2 | Planos de corte + trazas | ¿Cómo cambia S si fijo una variable? | Trazas y secciones transversales (§7.7) |
| 3 | Tazón con x centrada | ¿Por qué el valle es tan estrecho? | Hessiano, valores propios, cambio de variables |
| 4 | Valle + descenso por gradiente | ¿Cómo se llega al mínimo? | Gradiente, ∇S = 0, métodos numéricos |

---

## 8. Cómo leer los resultados del caso real

1. **La recta:** ŷ = 3.45x − 87.52. Cada pulgada adicional implica 3.45 lb más de peso promedio (≈ 0.62 kg por cm).
2. **El mínimo es seguro:** H definida positiva (det H = 16 800 > 0) ⇒ mínimo global único.
3. **Ajuste muy alto, pero no perfecto:** R² = 0.991 y RMSE = 1.42 lb (1.04 % del peso medio, 2.9 % del rango de 49 lb).
   Un R² tan alto se explica en parte porque los datos son **promedios**.
4. **Hay curvatura:** residuos en U (corr = 0.97, 3 rachas) y el modelo cuadrático reduce S de 30.23 a 1.77
   (F = 193, p ≈ 10⁻⁸). La relación real crece un poco más rápido para estaturas altas.
5. **Utilidad práctica:** las predicciones lineal y cuadrática difieren como mucho 2.5 lb, así que la recta es útil para
   estimar pesos promedio dentro del rango 58–72 in. **No debe extrapolarse** fuera de ese rango.
6. **Robustez:** los puntos extremos (x = 58 y 72) son los más influyentes; un error de 1 lb en un dato cambia la
   pendiente como mucho 0.025 lb/in.

---

## 9. Guion sugerido para la sustentación

| Paso | Vista | Qué mostrar | Qué decir |
|---|---|---|---|
| 1 | 1 | Dispersión | "La tendencia es aproximadamente lineal; proponemos y = ax + b." |
| 2 | 2 | Derivadas y sistema | "Minimizamos S(a, b): ∇S = 0 da las ecuaciones normales; a* = 3.45, b* = −87.52." |
| 3 | 6 | Criterio con ✓ | "H es definida positiva y constante: mínimo global." |
| 4 | 3 | Mover el deslizador a | "Cada recta es un punto; su error es la suma de los segmentos al cuadrado." |
| 5 | 4 | Gráfica nº 1 en log, girar | "S es un paraboloide; las curvas de nivel son elipses alrededor de (a*, b*)." |
| 6 | 4/5 | Gráfica nº 2 y trazas | "Las trazas son parábolas; solo coinciden en el mínimo." |
| 7 | 4 | Gráfica nº 3 (centrada) | "El valle es estrecho porque a y b están acoplados; centrando x es un tazón." |
| 8 | 4 | Gráfica nº 4 (gradiente) | "Bajando por el gradiente se llega al mismo punto donde ∇S = 0." |
| 9 | 8 | Tabla §9 | "Manual, código propio y bibliotecas coinciden; la diferencia es redondeo." |
| 10 | 7 y 9-C | Residuos en U y cuadrático | "La recta es muy buena, pero los residuos revelan una leve curvatura." |
