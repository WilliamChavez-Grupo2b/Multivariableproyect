# Cálculo manual con los datos reales (§5, §7.1–7.4)

> Este procedimiento se hace **a mano** (en el cuaderno) antes de usar el computador.
> El programa solo lo verifica (ver `analisis_resultados.md`, tabla de comparación).
> La deducción general de las fórmulas está en `desarrollo_matematico.md`.

## 1. Problema y datos (§5)

**Problema real:** ¿cómo cambia el peso promedio de una mujer adulta (30–39 años) con su estatura?

| Variable | Significado | Unidad |
|---|---|---|
| x | Estatura (variable independiente) | pulgadas (in) |
| y | Peso promedio para esa estatura (variable dependiente) | libras (lb) |

Fuente: *The World Almanac and Book of Facts* (1975), dataset `women` de R (ver `data/input/FUENTE.md`).
Conversión: 1 in = 2.54 cm, 1 lb = 0.4536 kg.

En el diagrama de dispersión (`results/images/01_dispersion.png`) los 15 puntos suben casi en línea
recta (r = 0.9955), así que el modelo **y = ax + b** es razonable como primera aproximación.

## 2. Tabla de cálculo

| i | xᵢ | yᵢ | xᵢ² | xᵢyᵢ | yᵢ² |
|---|---|---|---|---|---|
| 1 | 58 | 115 | 3364 | 6670 | 13225 |
| 2 | 59 | 117 | 3481 | 6903 | 13689 |
| 3 | 60 | 120 | 3600 | 7200 | 14400 |
| 4 | 61 | 123 | 3721 | 7503 | 15129 |
| 5 | 62 | 126 | 3844 | 7812 | 15876 |
| 6 | 63 | 129 | 3969 | 8127 | 16641 |
| 7 | 64 | 132 | 4096 | 8448 | 17424 |
| 8 | 65 | 135 | 4225 | 8775 | 18225 |
| 9 | 66 | 139 | 4356 | 9174 | 19321 |
| 10 | 67 | 142 | 4489 | 9514 | 20164 |
| 11 | 68 | 146 | 4624 | 9928 | 21316 |
| 12 | 69 | 150 | 4761 | 10350 | 22500 |
| 13 | 70 | 154 | 4900 | 10780 | 23716 |
| 14 | 71 | 159 | 5041 | 11289 | 25281 |
| 15 | 72 | 164 | 5184 | 11808 | 26896 |
| **Σ** | **975** | **2051** | **63655** | **134281** | **283803** |

n = 15, x̄ = 975/15 = **65**, ȳ = 2051/15 = **136.7333**

## 3. Función de error (§6)

$$S(a,b)=\sum_{i=1}^{15}(y_i-ax_i-b)^2$$

Expandiendo el cuadrado y usando las sumas de la tabla, S queda como un polinomio de grado 2 en a y b:

$$S(a,b)=\Big(\sum x_i^2\Big)a^2+2\Big(\sum x_i\Big)ab+n\,b^2-2\Big(\sum x_iy_i\Big)a-2\Big(\sum y_i\Big)b+\sum y_i^2$$

$$\boxed{S(a,b)=63655\,a^2+1950\,ab+15\,b^2-268562\,a-4102\,b+283803}$$

## 4. Derivadas parciales (§7.1)

$$\frac{\partial S}{\partial a}=-2\sum x_i(y_i-ax_i-b)=2\Big(a\sum x_i^2+b\sum x_i-\sum x_iy_i\Big)=127310\,a+1950\,b-268562$$

$$\frac{\partial S}{\partial b}=-2\sum (y_i-ax_i-b)=2\Big(a\sum x_i+nb-\sum y_i\Big)=1950\,a+30\,b-4102$$

(Se pueden comprobar derivando directamente el polinomio del paso 3.)

## 5. Punto crítico: ∇S = 0 (§7.2)

$$\nabla S=\begin{pmatrix}127310a+1950b-268562\\1950a+30b-4102\end{pmatrix}=\begin{pmatrix}0\\0\end{pmatrix}$$

Dividiendo entre 2 se obtienen las **ecuaciones normales**:

$$\begin{cases}63655\,a+975\,b=134281\\975\,a+15\,b=2051\end{cases}$$

## 6. Solución del sistema (§7.3)

De la segunda ecuación: $b=\dfrac{2051-975a}{15}$. Sustituyendo en la primera y multiplicando por 15:

$$15\cdot63655\,a+975(2051-975a)=15\cdot134281$$
$$954825\,a+1999725-950625\,a=2014215$$
$$4200\,a=14490\quad\Rightarrow\quad\boxed{a^*=3.45\ \text{lb/in}}$$

$$b^*=\frac{2051-3.45\cdot975}{15}=\frac{2051-3363.75}{15}=\frac{-1312.75}{15}\quad\Rightarrow\quad\boxed{b^*=-87.5167\ \text{lb}}$$

(Comprobación con la fórmula de medias: b* = ȳ − a*x̄ = 136.7333 − 3.45·65 = −87.5167 ✓)

**Recta de mejor ajuste:**

$$\boxed{\hat y=3.45\,x-87.5167}$$

**Interpretación:** por cada pulgada más de estatura, el peso promedio aumenta 3.45 lb
(≈ 1.56 kg por cada 2.54 cm, o 0.62 kg por cm). El intercepto b* = −87.5 lb **no tiene
significado físico**: sería el "peso" de una persona de 0 in, muy lejos del rango medido
(58–72 in). Es solo el parámetro que ubica la recta.

## 7. Error mínimo

En el mínimo se cumple $S_{mín}=\sum y_i^2-a^*\sum x_iy_i-b^*\sum y_i$:

$$S_{mín}=283803-3.45(134281)+\frac{1312.75}{15}(2051)=283803-463269.45+179496.6833$$
$$\boxed{S_{mín}=30.2333\ \text{lb}^2}$$

⚠️ **Ojo con el redondeo:** si se usa b* = −87.5167 (redondeado) en esta fórmula se obtiene 30.30,
porque se restan números de ~460 000 y un error de 0.00003 en b se multiplica por 2051.
Por eso se usa la fracción exacta −1312.75/15. También se puede sumar la columna eᵢ² de la tabla
de residuos (`analisis_resultados.md`).

## 8. Hessiano (§7.4)

Segundas derivadas (son constantes, no dependen de a ni de b):

$$\frac{\partial^2S}{\partial a^2}=2\sum x_i^2=127310,\qquad \frac{\partial^2S}{\partial b^2}=2n=30,\qquad \frac{\partial^2S}{\partial a\,\partial b}=2\sum x_i=1950$$

$$H=\begin{pmatrix}127310 & 1950\\ 1950 & 30\end{pmatrix}$$

**Criterio de las segundas derivadas:**

1. $S_{aa}=127310>0$
2. $\det H=127310\cdot30-1950^2=3819300-3802500=\boxed{16800>0}$

Como $S_{aa}>0$ y $\det H>0$, H es **definida positiva** ⇒ (a*, b*) es un **mínimo local**.
Y como H es constante (S es cuadrática), la superficie es convexa en todo el plano:
el mínimo es **global y único**.

**Autovalores** (opcional, refuerza la conclusión): resolviendo $\lambda^2-(\text{tr}H)\lambda+\det H=0$:

$$\lambda^2-127340\lambda+16800=0\ \Rightarrow\ \lambda_1\approx0.1319,\quad\lambda_2\approx127339.87$$

Ambos positivos ✓. Su cociente $\kappa=\lambda_2/\lambda_1\approx9.65\times10^5$ indica que S es
casi un millón de veces más empinada en una dirección que en otra: las curvas de nivel son elipses
**extremadamente alargadas** (ver `interpretacion_geometrica.md`).
