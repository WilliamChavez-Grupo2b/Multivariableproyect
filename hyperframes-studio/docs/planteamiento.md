# Planteamiento 
Se tiene un conjunto de datos:
\[ (x_1,y_1),(x_2,y_2),\ldots,(x_n,y_n) \]
Cada pareja representa una observación.
•	\(x_i\): valor de la variable independiente.
•	\(y_i\): valor real observado.
•	\(n\): cantidad total de observaciones.
Se propone utilizar el modelo:
\[ \boxed{y=ax+b} \]
Donde:
Parámetro	Significado
\(a\)	Pendiente de la recta.
\(b\)	Intersección con el eje vertical.
\(x\)	Variable independiente.
\(y\)	Valor estimado por el modelo.
El problema consiste en encontrar los valores óptimos \(a^*\) y \(b^*\).
4.1. ¿Qué es el error cuadrático?
Para cada observación, el modelo calcula un valor estimado:
\[ \hat y_i=ax_i+b \]
El error o residuo es la diferencia entre el valor real y el estimado:
\[ e_i=y_i-\hat y_i \]
Reemplazando el modelo:
\[ \boxed{e_i=y_i-ax_i-b} \]
Este error puede ser positivo, negativo o cero.
Para evitar que los errores positivos y negativos se cancelen, se elevan al cuadrado:
\[ e_i^2=(y_i-ax_i-b)^2 \]
Finalmente, se suman todos los errores cuadrados:
\[ \boxed{ S(a,b)=\sum_{i=1}^{n}(y_i-ax_i-b)^2 } \]
Esta es la función que debemos minimizar.
¿Qué significa minimizarla?
Significa encontrar una recta que haga que la suma de las diferencias cuadradas entre los datos reales y los valores estimados sea lo más pequeña posible.
No buscamos que cada punto tenga necesariamente un error cero, sino que el error total sea mínimo.
