# Guía para la sustentación

La rúbrica da **40 %** a "Comprensión conceptual" y "Evaluación individual" (respuestas a preguntas).
Cada integrante debe poder responder **todas** estas preguntas sin leer.

## Orden sugerido de la presentación (≈ 12 min)

| # | Tema | Qué mostrar | Quién |
|---|---|---|---|
| 1 | Problema y datos | Tabla, unidades, fuente, `01_dispersion.png` | |
| 2 | Modelo y S(a, b) | Residuo eᵢ, por qué se eleva al cuadrado | |
| 3 | Cálculo a mano | Derivadas, ∇S = 0, sistema, a*, b* (en el tablero) | |
| 4 | Hessiano | H, det H = 16800 > 0 ⇒ mínimo | |
| 5 | Geometría | `modelos_3d.html`: superficie, trazas, curvas de nivel, explorador | |
| 6 | Código | Pseudocódigo + tabla manual / código / biblioteca | |
| 7 | Recta, residuos y calidad | `02`, `06`, `07`, R² y RMSE interpretados | |
| 8 | Análisis adicional | `10_influencia.png`, `11_lineal_vs_cuadratico.png` | |
| 9 | Conclusiones | Responder la pregunta central | |

**Consejo:** abran `results/html/modelos_3d.html` y usen el **explorador (modelo 3)** en vivo: mover
el deslizador y ver que la recta óptima es el fondo del tazón es la forma más clara de mostrar la
interpretación geométrica.

## Preguntas probables y respuestas

### Sobre el planteamiento

**¿Por qué S(a, b) es una función de dos variables si los datos tienen x e y?**
Porque los datos (xᵢ, yᵢ) son números fijos. Lo que podemos cambiar es la recta, es decir, a y b.
Cada par (a, b) es una recta distinta y S mide su error total.

**¿Por qué se elevan los residuos al cuadrado y no se suman tal cual?**
Porque los positivos y los negativos se cancelarían: una recta muy mala podría tener Σeᵢ = 0.
Al elevarlos al cuadrado todos cuentan como positivos y los errores grandes pesan más. Además,
S queda derivable, lo que no pasa con |eᵢ|.

**¿Qué significan a* y b* en el problema?**
a* = 3.45 lb/in: por cada pulgada más de estatura, el peso promedio sube 3.45 lb (≈ 0.62 kg/cm).
b* = −87.5 lb no tiene sentido físico (el peso para estatura 0). Solo ubica la recta, y muestra
que no se debe extrapolar.

### Sobre el cálculo

**¿Cómo se deriva S respecto de a?**
Con la regla de la cadena: la derivada de (yᵢ − axᵢ − b)² respecto de a es 2(yᵢ − axᵢ − b)·(−xᵢ).
Sumando: ∂S/∂a = −2Σxᵢ(yᵢ − axᵢ − b).

**¿Qué significa ∇S = 0 geométricamente?**
Que el plano tangente a la superficie es horizontal: no hay ninguna dirección en la que S suba o
baje en primera aproximación. Es la condición necesaria para un mínimo.

**¿Por qué ∇S = 0 no basta?**
Porque también se cumple en máximos y puntos de silla. Para saber cuál es hay que mirar las
segundas derivadas, es decir, el Hessiano.

**¿Qué dice el criterio del Hessiano aquí?**
Sₐₐ = 2Σx² = 127310 > 0 y det H = 4[nΣx² − (Σx)²] = 16800 > 0 ⇒ H es definida positiva ⇒ el punto
crítico es un mínimo.

**¿Por qué es mínimo global y no solo local?**
Porque H no depende de (a, b): es la misma en todo el plano. S es convexa en todas partes y por
eso tiene un solo mínimo.

**¿Cuándo fallaría el método?**
Si todos los xᵢ fueran iguales: entonces nΣx² − (Σx)² = 0, det H = 0 y no hay una recta única
(el sistema tiene infinitas soluciones: cualquier recta que pase por (x̄, ȳ) da el mismo error). Matemáticamente, nΣx² − (Σx)² = nΣ(xᵢ − x̄)².

**¿Por qué Σeᵢ = 0 exactamente?**
Porque es la ecuación ∂S/∂b = 0: −2Σeᵢ = 0. Del mismo modo, ∂S/∂a = 0 da Σxᵢeᵢ = 0.

### Sobre la geometría

**¿Qué forma tiene la superficie y por qué?**
Es un paraboloide elíptico: S es un polinomio de grado 2 en (a, b) con H constante y definida
positiva. **No puede tener ondulaciones ni varios mínimos.**

**¿Qué son las trazas y qué muestran?**
Son los cortes de la superficie con planos a = constante o b = constante. Aquí son parábolas que
abren hacia arriba. Si fuera un punto de silla, una abriría hacia abajo.

**¿Por qué las curvas de nivel son elipses tan alargadas?**
Porque los autovalores de H son muy distintos (0.13 y 127340, κ ≈ 10⁶). S cambia muchísimo si
movemos la pendiente y muy poco si movemos a la vez a y b a lo largo del valle b ≈ ȳ − a·x̄. Esto
pasa porque x está lejos de 0 (x̄ = 65): cambiar la pendiente se puede compensar con el intercepto.

**¿Qué pasa si centramos x?**
Con u = x − x̄, Σu = 0, el término cruzado de H desaparece y las elipses quedan alineadas con los
ejes, casi círculos. El mínimo es el mismo (b* = c* − a*·x̄).

### Sobre el código

**¿El programa usa una función de regresión ya hecha?**
No para el ajuste principal. `resolver_analitico` programa las fórmulas que deducimos a mano.
`polyfit` y `linregress` se usan **después**, solo para comparar, como pide el enunciado.

**¿Por qué el resultado manual difiere un poco del código?**
Por el redondeo a 4 decimales hecho a mano (b* = −1312.75/15 es periódico). La diferencia es 3·10⁻⁵.
Entre el código y la biblioteca la diferencia es ~10⁻¹⁴ (punto flotante).

**¿Por qué no converge el descenso por gradiente?**
Por la forma de valle (κ ≈ 10⁶): el paso debe ser pequeño para no salirse en la dirección
empinada, y con ese paso avanza casi nada por el valle. Centrando x converge en 550 iteraciones, y
Newton (que usa H) llega en 1 paso.

### Sobre los resultados

**R² = 0.991, ¿entonces el modelo es perfecto?**
No. Los residuos forman una U (positivos en los extremos, negativos en el centro): es una
desviación sistemática, porque la relación es un poco curva. R² alto solo dice que la recta explica
mucha variación. Además, los datos son promedios, lo que sube el R².

**¿Qué significa RMSE = 1.42 lb?**
Es el error típico de la recta al predecir el peso promedio: alrededor del 1 % del peso medio. Para
estimar dentro de 58–72 in es pequeño.

**¿Qué dato influye más?**
x = 72 in: si se quita, la pendiente baja de 3.45 a 3.35. Los extremos influyen más porque están
lejos de x̄ y la pendiente pondera cada dato por (xᵢ − x̄).

**¿Es mejor el modelo cuadrático?**
Reduce S de 30.2 a 1.8 y elimina el patrón de los residuos (prueba F: p ≈ 10⁻⁸): la curvatura es
real. Pero las predicciones solo difieren hasta 2.5 lb. Para interpretar, la recta es más simple
y suficiente; para describir con precisión, la cuadrática es mejor.

**¿Cuál es la respuesta a la pregunta central?**
El Cálculo Multivariable convierte "la mejor recta" en "el mínimo de S(a, b)". ∇S = 0 lo encuentra,
el Hessiano garantiza que es un mínimo global, la geometría (paraboloide, trazas, curvas de nivel)
lo hace visible, y el computador lo verifica y permite analizar su calidad y sus limitaciones.
