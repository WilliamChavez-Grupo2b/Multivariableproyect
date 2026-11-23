# Proyecto de Cálculo Multivariable — Regresión lineal y optimización

**Pregunta central:** ¿cómo se puede utilizar el Cálculo Multivariable para determinar y analizar
el modelo lineal que mejor representa un conjunto de datos?

**Problema real:** estatura (in) vs. peso promedio (lb) de mujeres estadounidenses de 30–39 años
(15 datos, *World Almanac* 1975). Ver [`FUENTE.md`](hyperframes-studio/data/input/FUENTE.md).

**Resultado:** ŷ = 3.45·x − 87.5167, S_mín = 30.2333 lb², R² = 0.991, RMSE = 1.42 lb.

## Para la sustentación

- 🧊 **Modelos 3D interactivos:** abrir `hyperframes-studio/results/html/modelos_3d.html` en el
  navegador (funciona sin internet). Superficie con trazas y curvas de nivel, el "tazón" con x
  centrada, un **explorador** que mueve la recta y su punto sobre la superficie, y los métodos
  numéricos en 3D.
- ❓ **Preguntas probables y respuestas:** `hyperframes-studio/docs/guia_sustentacion.md`.

## Dónde está cada parte del enunciado

| Sección del enunciado | Documento / figura |
|---|---|
| §5 Exploración de datos | `docs/calculo_manual.md` §1–2, `01_dispersion.png` |
| §6 Construcción del modelo | `docs/planteamiento.md` |
| §7.1–7.4 Derivadas, ∇S = 0, solución, Hessiano (a mano) | `docs/calculo_manual.md`, `docs/desarrollo_matematico.md` |
| §7.5–7.8 Superficie, trazas, curvas de nivel | `docs/interpretacion_geometrica.md`, figuras `03`–`05`, `results/html/modelos_3d.html` |
| §8 Verificación computacional | `src/` y `docs/analisis_resultados.md` §8 |
| §9 Tabla analítico / código propio / biblioteca | `docs/analisis_resultados.md` §9 |
| §10–12 Recta, residuos, calidad | `docs/analisis_resultados.md`, figuras `02`, `06`, `07` |
| §13 Análisis adicional (A, B y C) | `docs/analisis_resultados.md` §13, figuras `10`, `11` |
| Conclusiones | `docs/Conclusiones.md` |
| Pseudocódigo y diagrama de flujo | `docs/pseudocodigo.md` |

Las figuras están en `hyperframes-studio/results/images/`.

## Cómo ejecutar

```bash
cd hyperframes-studio
python -m pip install numpy matplotlib scipy plotly openpyxl imageio imageio-ffmpeg pillow
python src/main.py --sin-gui     # todo el análisis + results/
python tests/pruebas.py          # 21 pruebas automáticas
```

`src/generar_datos.py` solo crea datos **sintéticos** para pruebas (`datos_sinteticos.csv`).
No modifica los datos reales.
