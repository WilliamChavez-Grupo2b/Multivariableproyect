Los valores \(a^*\) y \(b^*\) corresponden a un mínimo global de la función de error cuadrático, siempre que existan al menos dos valores diferentes de \(x\).
Esto es importante porque demuestra matemáticamente que no encontramos simplemente cualquier punto crítico, sino la combinación de parámetros que produce el menor error cuadrático total.
Si todos los valores de \(x\) son iguales, la pendiente y el intercepto no pueden determinarse de manera única mediante estos datos.
7. ¿Qué sucede gráficamente con los valores?
Esta parte es muy importante para interpretar correctamente el ejercicio.
Cuando representamos los datos en un plano cartesiano, obtenemos una serie de puntos:
\[ (x_1,y_1),(x_2,y_2),\ldots,(x_n,y_n) \]
Cada punto representa una observación real.
La recta:
\[ \hat y=a^*x+b^* \]
representa la tendencia lineal que mejor se ajusta a esos puntos según el criterio de mínimos cuadrados.
Ejemplo visual: relación positiva
Regression Modelling for Biostatistics 1 - 1  Simple Linear Regression
bookdown.org
Los puntos tienden a subir de izquierda a derecha. La pendiente es positiva.
Ejemplo visual: relación negativa
Correlation and Regression
2012books.lardbucket.org
Los puntos tienden a bajar de izquierda a derecha. La pendiente es negativa.
Ejemplo visual: relación lineal débil o inexistente
Regressione lineare semplice
webtutordimatematica.it
Los puntos no muestran una tendencia lineal clara. La pendiente puede estar cerca de cero.
7.1. ¿Qué significa que los valores se atraigan o se repelan?
Hay que aclarar algo importante: en este ejercicio no existen fuerzas físicas de atracción o repulsión. No estamos estudiando cargas eléctricas ni partículas.
Lo que estamos estudiando es cómo se relacionan dos variables.
Sin embargo, podemos interpretar el comportamiento de los puntos mediante tres situaciones:
Caso 1. Relación positiva: los valores aumentan juntos
Cuando la pendiente es positiva:
\[ a^*>0 \]
Significa que cuando aumenta \(x\), generalmente también aumenta \(y\).
Por ejemplo:
•	A mayor cantidad de horas de estudio, generalmente puede haber mayor rendimiento académico.
•	A mayor cantidad de productos vendidos, mayores pueden ser los ingresos.
En la gráfica, la recta sube de izquierda a derecha.
Caso 2. Relación negativa: los valores se comportan en sentido contrario
Cuando la pendiente es negativa:
\[ a^*<0 \]
Significa que cuando aumenta \(x\), generalmente disminuye \(y\).
Por ejemplo:
•	A mayor velocidad de un vehículo, menor puede ser el tiempo necesario para recorrer una distancia fija.
•	A mayor precio de un producto, la cantidad demandada puede disminuir.
En la gráfica, la recta baja de izquierda a derecha.
Esto no significa que las variables se repelan físicamente, sino que presentan una relación inversa.
Caso 3. Relación lineal débil o inexistente
Cuando la pendiente es aproximadamente cero:
\[ a^*\approx 0 \]
Significa que no se observa una tendencia lineal importante entre las variables.
La recta será aproximadamente horizontal.
Esto no significa necesariamente que las variables no tengan ninguna relación: podrían presentar una relación curva u otro comportamiento que el modelo lineal no logra representar.
7.2. ¿Qué sucede con los puntos que están por encima y por debajo de la recta?
Aquí aparece otro concepto fundamental: los residuos.
Recordemos:
\[ e_i=y_i-\hat y_i \]
Existen tres posibilidades:
Situación	Resultado	Interpretación
Punto por encima de la recta	\(e_i>0\)	El valor real es mayor que el estimado.
Punto por debajo de la recta	\(e_i<0\)	El valor real es menor que el estimado.
Punto sobre la recta	\(e_i=0\)	El valor real coincide con el estimado.
¿Por qué es importante?
Porque la función de error cuadrático intenta reducir las distancias verticales entre los puntos reales y la recta.
Por ejemplo, si un punto está muy alejado de la recta, su error al cuadrado será considerablemente mayor.
Por eso, el método de mínimos cuadrados busca una recta que equilibre los errores de todas las observaciones.
8. Verificación mediante herramientas computacionales
Después de realizar todo el procedimiento matemático, podemos comprobar los resultados mediante programación en Python.
La computadora permite:
•	Introducir los datos experimentales.
•	Calcular la pendiente óptima.
•	Calcular el intercepto.
•	Obtener la ecuación de regresión.
•	Calcular los errores individuales.
•	Calcular el error cuadrático total.
•	Dibujar los puntos y la recta.
•	Analizar si existe una relación positiva, negativa o débil.
