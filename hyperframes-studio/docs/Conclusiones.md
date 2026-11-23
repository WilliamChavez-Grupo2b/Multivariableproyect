# Conclusiones

## Respuesta a la pregunta central

> *¿Cómo se puede utilizar el Cálculo Multivariable para determinar y analizar el modelo lineal
> que mejor representa un conjunto de datos?*

Convirtiendo la pregunta "¿cuál es la mejor recta?" en un **problema de optimización de una función
de dos variables**. Cada recta y = ax + b es un punto (a, b) del plano, y su error total
S(a, b) = Σ(yᵢ − axᵢ − b)² es la "altura" de ese punto. La mejor recta es el **punto más bajo de la
superficie z = S(a, b)**, y el Cálculo Multivariable da las herramientas para encontrarlo y
justificarlo:

1. **Analítico:** las derivadas parciales y la condición ∇S = 0 dan un sistema lineal de 2×2
   (las ecuaciones normales). Resuelto a mano con los 15 datos reales da
   **a* = 3.45 lb/in y b* = −87.5167 lb**, con S_mín = 30.2333 lb².
2. **Hessiano:** H = [[127310, 1950], [1950, 30]] tiene Sₐₐ > 0 y det H = 16800 > 0. Es
   definido positivo y constante, así que (a*, b*) es el **mínimo global y único**, no un
   máximo ni un punto de silla.
3. **Geométrico:** S es un paraboloide elíptico. Sus trazas son parábolas que abren hacia arriba y
   sus curvas de nivel son elipses concéntricas cuyo centro es exactamente (a*, b*).
4. **Computacional:** el código propio (que programa las fórmulas deducidas), `numpy.polyfit` y
   `scipy.stats.linregress` dan los mismos valores (diferencias ~10⁻¹⁴). Con el cálculo manual la
   diferencia es ~10⁻⁵ y se debe solo al redondeo.

## Sobre el problema real (estatura y peso)

- Por cada **pulgada** adicional de estatura, el peso promedio de las mujeres de 30–39 años
  aumenta **3.45 lb** (≈ 0.62 kg por cada cm). La relación es positiva y muy fuerte
  (r = 0.995, R² = 0.991).
- El error típico de la recta es **RMSE = 1.42 lb**, alrededor del 1 % del peso medio. Para
  estimar el peso promedio a partir de la estatura **dentro del rango medido (58–72 in)**, la recta
  es útil.
- El intercepto b* = −87.5 lb no tiene interpretación física. Muestra que la recta **no debe
  extrapolarse** lejos de los datos.

## Reflexión crítica: limitaciones del modelo lineal

- **R² alto no significa modelo correcto.** Los residuos tienen un patrón en **U**: positivos en los
  extremos y negativos en el centro. La recta sobreestima el peso en las estaturas medias y lo
  subestima en las bajas y altas: hay una **desviación sistemática**.
- El modelo cuadrático reduce el error de S = 30.23 a S = 1.77, y la prueba F confirma que la
  curvatura es real. Sin embargo, las predicciones de ambos modelos difieren a lo sumo 2.5 lb. La
  elección depende del propósito: **la recta para interpretar y estimar, la cuadrática para
  describir la forma exacta.**
- Los datos son **promedios** (no personas individuales) y de 1975, de una población específica.
  El modelo no describe a una persona concreta, que puede estar muy lejos del promedio, ni a
  poblaciones actuales o de otros países.
- Los puntos de los extremos (58 y 72 in) son los más influyentes. Quitar x = 72 cambia la
  pendiente un 3 %. El ajuste es robusto, pero depende más de los extremos que del centro.

## Lo que aprendimos de la geometría

- La forma de la superficie la determina el Hessiano. Aquí sus autovalores difieren en un factor
  ~10⁶: S es un **valle largo y estrecho**, porque a y b están acoplados (aumentar a se puede
  compensar bajando b).
- Esa forma explica por qué el descenso por gradiente simple **no logra converger** en 200 000
  iteraciones, mientras que centrar x (que vuelve diagonal a H) lo hace converger en 550 y el
  método de Newton (que usa H) llega en **un solo paso**.
- La solución analítica ∇S = 0 no tiene ese problema: da el mínimo **exacto** sin importar la
  forma del valle. El computador no reemplazó el desarrollo matemático. Sirvió para verificarlo,
  visualizarlo y descubrir sus limitaciones.
