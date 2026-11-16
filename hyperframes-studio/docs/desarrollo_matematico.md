# Desarrollo matematico
Aquí comienza la parte matemática principal del ejercicio.
Tenemos la función:
\[ S(a,b)=\sum_{i=1}^{n}(y_i-ax_i-b)^2 \]
Como la función depende de dos variables, \(a\) y \(b\), utilizaremos derivadas parciales.
Paso 1. Derivar parcialmente respecto a \(a\)
La derivada parcial respecto a \(a\) consiste en derivar la función considerando que \(b\) permanece constante.
Partimos de:
\[ S(a,b)=\sum_{i=1}^{n}(y_i-ax_i-b)^2 \]
Aplicamos la regla de la cadena:
\[ \frac{\partial S}{\partial a} = \sum_{i=1}^{n} 2(y_i-ax_i-b)(-x_i) \]
El factor \(-x_i\) aparece porque la derivada de:
\[ y_i-ax_i-b \]
respecto a \(a\) es:
\[ -x_i \]
Por lo tanto:
\[ \frac{\partial S}{\partial a} = -2\sum_{i=1}^{n}x_i(y_i-ax_i-b) \]
Para encontrar el punto crítico, igualamos a cero:
\[ -2\sum_{i=1}^{n}x_i(y_i-ax_i-b)=0 \]
Dividimos entre \(-2\):
\[ \sum_{i=1}^{n}x_i(y_i-ax_i-b)=0 \]
Distribuimos \(x_i\):
\[ \sum_{i=1}^{n}x_iy_i -a\sum_{i=1}^{n}x_i^2 -b\sum_{i=1}^{n}x_i=0 \]
Organizamos:
\[ \boxed{ a\sum x_i^2+b\sum x_i=\sum x_iy_i } \]
Esta es nuestra primera ecuación normal.
Paso 2. Derivar parcialmente respecto a \(b\)
Ahora derivamos la misma función, pero respecto a \(b\), considerando \(a\) constante.
\[ \frac{\partial S}{\partial b} = \sum_{i=1}^{n} 2(y_i-ax_i-b)(-1) \]
Simplificamos:
\[ \frac{\partial S}{\partial b} = -2\sum_{i=1}^{n}(y_i-ax_i-b) \]
Igualamos a cero:
\[ -2\sum_{i=1}^{n}(y_i-ax_i-b)=0 \]
Dividimos entre \(-2\):
\[ \sum_{i=1}^{n}(y_i-ax_i-b)=0 \]
Desarrollamos:
\[ \sum y_i-a\sum x_i-b\sum 1=0 \]
Como existen \(n\) observaciones:
\[ \sum 1=n \]
Entonces:
\[ \sum y_i-a\sum x_i-nb=0 \]
Organizamos:
\[ \boxed{ a\sum x_i+nb=\sum y_i } \]
Esta es nuestra segunda ecuación normal.
Paso 3. Formar el sistema de ecuaciones
Hemos obtenido dos ecuaciones:
Primera:
\[ a\sum x_i^2+b\sum x_i=\sum x_iy_i \]
Segunda:
\[ a\sum x_i+nb=\sum y_i \]
Las escribimos como un sistema:
\[ \boxed{ \begin{cases} a\sum x_i^2+b\sum x_i=\sum x_iy_i\\ a\sum x_i+nb=\sum y_i \end{cases} } \]
Ahora debemos resolver este sistema para encontrar \(a^*\) y \(b^*\).
Paso 4. Despejar el valor óptimo de \(a\)
De la segunda ecuación:
\[ a\sum x_i+nb=\sum y_i \]
Despejamos \(b\):
\[ nb=\sum y_i-a\sum x_i \]
\[ b=\frac{\sum y_i-a\sum x_i}{n} \]
Sustituimos esta expresión en la primera ecuación:
\[ a\sum x_i^2+ \left(\frac{\sum y_i-a\sum x_i}{n}\right) \sum x_i = \sum x_iy_i \]
Multiplicamos todo por \(n\):
\[ na\sum x_i^2+ \sum x_i\sum y_i -a\left(\sum x_i\right)^2 = n\sum x_iy_i \]
Agrupamos los términos que contienen \(a\):
\[ a\left[ n\sum x_i^2-\left(\sum x_i\right)^2 \right] = n\sum x_iy_i-\sum x_i\sum y_i \]
Finalmente, despejamos:
\[ \boxed{ a^*= \frac{ n\sum x_iy_i-\left(\sum x_i\right)\left(\sum y_i\right) }{ n\sum x_i^2-\left(\sum x_i\right)^2 } } \]
Este resultado representa la pendiente óptima de la recta.
Paso 5. Encontrar el valor óptimo de \(b\)
Recordamos que:
\[ b=\frac{\sum y_i-a\sum x_i}{n} \]
Sustituimos el valor de \(a^*\):
\[ \boxed{ b^*=\frac{\sum y_i-a^*\sum x_i}{n} } \]
También podemos expresarlo utilizando las medias:
\[ \boxed{b^*=\bar y-a^*\bar x} \]
Donde:
\[ \bar x=\frac{\sum x_i}{n} \]
\[ \bar y=\frac{\sum y_i}{n} \]
Por lo tanto, ya encontramos los dos parámetros que minimizan la función.
Resultados matemáticos
Pendiente óptima
\[ \boxed{ a^*= \frac{n\sum x_iy_i-\sum x_i\sum y_i} {n\sum x_i^2-(\sum x_i)^2} } \]
Intercepto óptimo
\[ \boxed{b^*=\bar y-a^*\bar x} \]
Recta de regresión final
\[ \boxed{\hat y=a^*x+b^*} \]
6. Comprobación mediante la matriz Hessiana
Después de encontrar los puntos críticos, debemos comprobar si realmente corresponden a un mínimo.
Para esto utilizamos la matriz Hessiana, que contiene las segundas derivadas parciales de la función.
Nuestra función es:
\[ S(a,b)=\sum_{i=1}^{n}(y_i-ax_i-b)^2 \]
Paso 1. Calcular la segunda derivada respecto a \(a\)
Ya sabemos que:
\[ \frac{\partial S}{\partial a} =-2\sum x_i(y_i-ax_i-b) \]
Derivamos nuevamente respecto a \(a\):
\[ \frac{\partial^2 S}{\partial a^2} = 2\sum_{i=1}^{n}x_i^2 \]
Paso 2. Calcular las derivadas cruzadas
Derivamos la primera derivada respecto a \(b\):
\[ \frac{\partial^2 S}{\partial a\partial b} = 2\sum_{i=1}^{n}x_i \]
Ahora derivamos la segunda derivada respecto a \(b\), respecto a \(b\) nuevamente:
\[ \frac{\partial^2 S}{\partial b^2}=2n \]
Por lo tanto, la matriz Hessiana es:
\[ \boxed{ H= \begin{pmatrix} 2\sum x_i^2 & 2\sum x_i\\ 2\sum x_i & 2n \end{pmatrix} } \]
Paso 3. Calcular el determinante
Para comprobar que tenemos un mínimo, necesitamos que la matriz Hessiana sea definida positiva.
Una matriz simétrica de segundo orden es definida positiva cuando:
1.	Su primer elemento diagonal es positivo.
2.	Su determinante es positivo.
Primera condición:
\[ 2\sum x_i^2>0 \]
Esto se cumple siempre que no todos los valores de \(x_i\) sean cero.
Segunda condición:
\[ \det(H)>0 \]
Calculamos:
\[ \det(H)= \begin{vmatrix} 2\sum x_i^2 & 2\sum x_i\\ 2\sum x_i & 2n \end{vmatrix} \]
Multiplicamos los elementos de la diagonal principal y restamos el producto de la diagonal secundaria:
\[ \det(H)= 4n\sum x_i^2-4\left(\sum x_i\right)^2 \]
Sacamos factor común:
\[ \boxed{ \det(H)=4\left[ n\sum x_i^2-\left(\sum x_i\right)^2 \right] } \]
Esta expresión es positiva cuando los valores de \(x\) no son todos iguales.
Por tanto:
\[ \boxed{\det(H)>0} \]
Y la matriz Hessiana es definida positiva.
