# Fuente de los datos

**Problema real:** relación entre la **estatura** y el **peso promedio** de mujeres
estadounidenses de 30 a 39 años.

| Variable | Significado | Unidad |
|---|---|---|
| `x` | Estatura | pulgadas (in) — 1 in = 2.54 cm |
| `y` | Peso promedio para esa estatura | libras (lb) — 1 lb = 0.4536 kg |

- **Fuente original:** *The World Almanac and Book of Facts*, 1975 (datos que, según la
  documentación, provienen del *Build and Blood Pressure Study* de la Society of Actuaries).
- **Referencia:** McNeil, D. (1977). *Interactive Data Analysis: A Practical Primer*. Wiley.
- **Acceso:** conjunto `women` del paquete `datasets` de R
  (https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/women.html,
  CSV: https://vincentarelbundock.github.io/Rdatasets/csv/datasets/women.csv).
- **Observaciones:** n = 15. Los pesos se tomaron con ropa ligera de interior y zapatos,
  y las estaturas con zapatos. Cada `y` es un **promedio**, no una persona individual.

Se conservan las unidades originales (enteros) para que el cálculo manual sea sencillo.
