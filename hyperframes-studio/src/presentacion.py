"""
presentacion.py
---------------
Genera la PRESENTACIÓN INTERACTIVA de la sustentación:

    results/html/presentacion.html   (un solo archivo, funciona sin internet)

Organizada según los criterios de evaluación del profesor:
  Comprensión conceptual 20 % · Evaluación individual 20 % · Gráficos y análisis 20 %
  Presentación 15 % · Conclusiones 15 % · Programación y lógica 10 %

Controles: ← → (o espacio) para avanzar · N muestra las notas del expositor ·
I muestra el índice · F pantalla completa · en la diapositiva de la solución,
el botón "Siguiente paso" revela el procedimiento.

Ejecutar:  python src/presentacion.py     (también lo llama main.py)
"""

import base64
import json
import os
import sys

import numpy as np
import plotly.offline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datos import obtener_datos, PROBLEMA, FUENTE
from analitico import reporte_analitico
from modelo import error_cuadratico
import analisis
import modelos_3d
from gradiente import descenso_gradiente, metodo_newton

# Escriban aquí sus nombres (aparecen en la portada)
INTEGRANTES = ["Integrante 1", "Integrante 2", "Integrante 3", "Integrante 4", "Integrante 5"]
CURSO = "Cálculo Multivariable — Universidad Cooperativa de Colombia"

RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
IMG = os.path.join(RAIZ, "results", "images")
SALIDA = os.path.join(RAIZ, "results", "html", "presentacion.html")

CRITERIOS = {
    "conc": ("Comprensión conceptual", "20 %", "#0B5FCC"),
    "ind": ("Evaluación individual", "20 %", "#8E44AD"),
    "graf": ("Gráficos y análisis", "20 %", "#1B8A4C"),
    "pres": ("Presentación", "15 %", "#5A6472"),
    "concl": ("Conclusiones", "15 %", "#D62728"),
    "prog": ("Programación y lógica", "10 %", "#E8740C"),
}


def img(nombre, alto="62vh"):
    with open(os.path.join(IMG, nombre), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-height:{alto}" alt="{nombre}">'


def plot_div(fig, alto=560):
    fig.update_layout(height=alto, autosize=True)
    return plotly.offline.plot(fig, include_plotlyjs=False, output_type="div",
                               config={"displaylogo": False, "responsive": True})


def f4(v):
    return f"{v:.4f}".replace("-", "−")


def construir():
    x, y = obtener_datos()
    n = len(x)
    r = reporte_analitico(x, y)
    a, b, smin = r["a_star"], r["b_star"], r["s_min"]
    hv = analisis.verificar_hessiano_numerico(x, y, a, b)
    comp = analisis.comparar_metodos(x, y)
    res = analisis.analizar_residuos(x, y, a, b)
    cal = analisis.calidad_ajuste(x, y, a, b)
    inf = analisis.influencia_datos(x, y)
    cq = analisis.comparar_cuadratico(x, y)
    sen = analisis.sensibilidad(x, y)
    gd_s = descenso_gradiente(x, y)
    gd_c = descenso_gradiente(x, y, centrar=True)
    nw = metodo_newton(x, y)
    H = hv["H"]; lam = hv["autovalores"]
    xm, ym = x.mean(), y.mean()
    r_corr = float(np.corrcoef(x, y)[0, 1])
    e = res["e"]

    # ── tabla de datos ────────────────────────────────────────────────────
    filas = "".join(
        f"<tr><td>{i+1}</td><td>{xi:.0f}</td><td>{yi:.0f}</td><td>{xi*xi:.0f}</td>"
        f"<td>{xi*yi:.0f}</td><td>{yi*yi:.0f}</td></tr>" for i, (xi, yi) in enumerate(zip(x, y)))
    filas += (f"<tr class='tot'><td>Σ</td><td>{r['sum_x']:.0f}</td><td>{r['sum_y']:.0f}</td>"
              f"<td>{r['sum_x2']:.0f}</td><td>{r['sum_xy']:.0f}</td><td>{np.sum(y*y):.0f}</td></tr>")

    filas_res = "".join(
        f"<tr><td>{xi:.0f}</td><td>{yi:.0f}</td><td>{a*xi+b:.2f}</td>"
        f"<td class='{'pos' if ei > 0 else 'neg'}'>{ei:+.2f}</td></tr>"
        for xi, yi, ei in zip(x, y, e))

    filas_comp = "".join(
        f"<tr><td>{nom}</td><td>{m:.4f}</td><td>{cod:.10f}</td><td>{lib:.10f}</td><td>{d3:.1e}</td></tr>"
        for nom, m, cod, lib, d3, _ in comp["filas"])

    filas_inf = "".join(
        f"<tr{' class=destacada' if f is inf['mas_influyente'] else ''}><td>{f['x']:.0f}</td>"
        f"<td>{f['a']:.4f}</td><td>{f['da']:+.4f}</td><td>{f['b']:.2f}</td><td>{f['db']:+.2f}</td></tr>"
        for f in inf["filas"] if f["x"] in (58, 59, 64, 65, 66, 71, 72))

    # ── figuras interactivas ──────────────────────────────────────────────
    fig_sup = modelos_3d.modelo_superficie(x, y, a, b, smin)
    fig_taz = modelos_3d.modelo_tazon(x, y, a, b, smin)
    fig_met = modelos_3d.modelo_metodos(x, y, a, b, gd_s, gd_c, nw)

    # datos para el explorador (JS): curvas de nivel en (a, c), c = b + a·x̄
    u = x - xm; c_s = b + a * xm
    da = 0.6; dc = 6.0
    a_g = np.linspace(a - da, a + da, 121); c_g = np.linspace(c_s - dc, c_s + dc, 121)
    A, C = np.meshgrid(a_g, c_g)
    Sg = ((y[None, None, :] - A[..., None] * u[None, None, :] - C[..., None]) ** 2).sum(-1)
    explorador = {
        "x": x.tolist(), "y": y.tolist(), "xm": xm, "a": a, "b": b, "c": c_s, "smin": smin,
        "ag": a_g.tolist(), "cg": c_g.tolist(), "S": np.round(Sg, 3).tolist(),
        "amin": a - da, "amax": a + da, "cmin": c_s - dc, "cmax": c_s + dc,
    }

    # ── preguntas de práctica ─────────────────────────────────────────────
    preguntas = [
        ("¿Por qué S es una función de dos variables si los datos tienen x e y?",
         "Porque los datos (xᵢ, yᵢ) son números fijos. Lo que cambiamos es la recta, es decir, a y b. Cada (a, b) es una recta y S mide su error total."),
        ("¿Por qué se elevan los residuos al cuadrado?",
         "Para que los positivos y negativos no se cancelen, para que los errores grandes pesen más y para que S sea derivable (|eᵢ| no lo es en 0)."),
        ("¿Cómo se deriva S respecto de a?",
         "Regla de la cadena: d/da (yᵢ − axᵢ − b)² = 2(yᵢ − axᵢ − b)(−xᵢ). Sumando: ∂S/∂a = −2Σxᵢ(yᵢ − axᵢ − b)."),
        ("¿Qué significa ∇S = 0 geométricamente?",
         "Que el plano tangente a la superficie es horizontal: en ese punto S no sube ni baja en ninguna dirección (a primer orden)."),
        ("¿Por qué ∇S = 0 no basta para asegurar un mínimo?",
         "Porque también se cumple en máximos y puntos de silla. Hay que revisar el Hessiano (segundas derivadas)."),
        ("¿Qué dice el criterio del Hessiano?",
         f"Sₐₐ = 2Σx² = {H[0,0]:.0f} > 0 y det H = 4[nΣx² − (Σx)²] = {hv['det']:.0f} > 0 ⇒ H definida positiva ⇒ mínimo."),
        ("¿Por qué el mínimo es global?",
         "Porque H es constante (no depende de a ni de b) y definida positiva: S es convexa en todo el plano y tiene un solo mínimo."),
        ("¿Cuándo no existe una recta única?",
         "Si todos los xᵢ son iguales: nΣx² − (Σx)² = nΣ(xᵢ − x̄)² = 0, det H = 0 y el sistema tiene infinitas soluciones."),
        ("¿Por qué Σeᵢ = 0 exactamente?",
         "Porque es la ecuación ∂S/∂b = −2Σeᵢ = 0. De igual forma ∂S/∂a = 0 da Σxᵢeᵢ = 0."),
        ("¿Qué forma tiene la superficie? ¿Puede ser ondulada?",
         "Es un paraboloide elíptico: S es un polinomio de grado 2 con H constante y definida positiva. No puede tener ondulaciones ni varios mínimos."),
        ("¿Qué son las trazas y qué muestran?",
         "Cortes de la superficie con planos a = cte o b = cte. Aquí son parábolas que abren hacia arriba; en un punto de silla una abriría hacia abajo."),
        ("¿Por qué las curvas de nivel son elipses tan alargadas?",
         f"Porque los autovalores de H son muy distintos ({lam[0]:.2f} y {lam[1]:.0f}, κ ≈ 10⁶): x está lejos de 0 y un cambio de pendiente se compensa con el intercepto."),
        ("¿El programa usa una función de regresión ya hecha?",
         "No para el ajuste principal: resolver_analitico programa las fórmulas deducidas a mano. polyfit y linregress solo se usan para comparar."),
        ("¿Por qué el resultado manual difiere un poco del código?",
         "Por el redondeo a 4 decimales a mano (b* = −1312.75/15 es periódico): diferencia 3·10⁻⁵. Código vs biblioteca: ~10⁻¹⁴, error de punto flotante."),
        ("¿Por qué no converge el descenso por gradiente simple?",
         "Por el valle estrecho (κ ≈ 10⁶): el paso debe ser pequeño para no salirse en la dirección empinada y entonces avanza casi nada por el valle. Centrando x converge; Newton llega en 1 paso."),
        ("R² = 0.991: ¿el modelo es perfecto?",
         "No. Los residuos tienen forma de U (desviación sistemática): la relación es un poco curva. R² alto solo indica que se explica mucha variación; además los datos son promedios."),
        ("¿Qué significa RMSE = 1.42 lb?",
         "Es el error típico de la recta al predecir el peso promedio: ≈ 1 % del peso medio. Para estimar dentro de 58–72 in es pequeño."),
        ("¿Qué dato influye más en la recta?",
         "x = 72 in: si se quita, la pendiente pasa de 3.45 a 3.35. Los extremos pesan más porque están lejos de x̄."),
        ("¿Es mejor el modelo cuadrático?",
         f"Reduce S de {cq['S_lineal']:.1f} a {cq['S_cuad']:.1f} y elimina el patrón de residuos (la curvatura es real), pero las predicciones difieren a lo sumo {cq['max_dif_pred']:.1f} lb: para interpretar, la recta es suficiente."),
        ("¿Qué significan a* y b* en el problema?",
         "a* = 3.45 lb/in: cada pulgada más de estatura suma 3.45 lb al peso promedio (≈ 0.62 kg/cm). b* no tiene sentido físico (estatura 0): no se debe extrapolar."),
        ("¿Cuál es la respuesta a la pregunta central?",
         "El Cálculo Multivariable convierte 'la mejor recta' en 'el mínimo de S(a, b)': ∇S = 0 lo encuentra, el Hessiano garantiza que es mínimo global, la geometría lo hace visible y el computador lo verifica y analiza."),
    ]
    html_preg = "".join(
        f"<details class='preg'><summary><b>{i+1}.</b> {q}</summary><p>{ans}</p></details>"
        for i, (q, ans) in enumerate(preguntas))

    # ── diapositivas ──────────────────────────────────────────────────────
    # cada una: (criterios, título, contenido, notas)
    D = []

    D.append((["pres"], "Portada", f"""
<div class="portada">
  <p class="curso">{CURSO}</p>
  <h1>Regresión lineal y optimización<br>mediante funciones de varias variables</h1>
  <div class="pregunta">¿Cómo se puede utilizar el Cálculo Multivariable para determinar y analizar
  el modelo lineal que mejor representa un conjunto de datos?</div>
  <p class="integrantes">{' · '.join(INTEGRANTES)}</p>
  <p class="ayuda">← → avanzar · <b>N</b> notas · <b>I</b> índice · <b>F</b> pantalla completa</p>
</div>""", "Presentar al grupo y leer la pregunta central. Decir que la respuesta integra tres perspectivas: analítica, geométrica y computacional."))

    filas_rub = ""
    # índices (desde 0) de las diapositivas que cubren cada criterio
    mapa = {"conc": [2, 3, 5, 6, 7, 8, 15], "ind": [6, 7, 8, 23],
            "graf": [4, 9, 10, 11, 12, 13, 16, 17, 18, 19], "pres": [0, 1],
            "concl": [18, 19, 21, 22], "prog": [14, 15, 20]}
    for k, (nom, pct, col) in CRITERIOS.items():
        enlaces = " ".join(f"<a href='#' data-ir='{i}'>{i+1}</a>" for i in mapa[k])
        filas_rub += f"<tr><td><span class='chip' style='background:{col}'>{nom}</span></td><td>{pct}</td><td>{enlaces}</td></tr>"
    D.append((["pres"], "Mapa de la presentación según la rúbrica", f"""
<table class="tabla rub"><tr><th>Criterio</th><th>Peso</th><th>Diapositivas (clic para ir)</th></tr>{filas_rub}</table>
<p class="nota">Cada diapositiva indica arriba a la derecha qué criterio cubre.</p>""",
              "Mostrar que la presentación cubre todos los criterios de evaluación."))

    D.append((["conc"], "Pregunta central y enfoque", f"""
<div class="tres">
 <div class="caja"><h3>1 · Analítico</h3><p>Derivadas parciales, ∇S = 0, solución del sistema y criterio del Hessiano, <b>a mano</b>.</p></div>
 <div class="caja"><h3>2 · Geométrico</h3><p>Superficie z = S(a, b), trazas, curvas de nivel: la mejor recta es el <b>fondo</b> de la superficie.</p></div>
 <div class="caja"><h3>3 · Computacional</h3><p>Código propio que programa las fórmulas, comparación con bibliotecas, gráficas y análisis.</p></div>
</div>
<p class="grande">Idea clave: cada recta y = ax + b es un <b>punto (a, b)</b> del plano, y su error total
S(a, b) es una <b>altura</b>. Buscar la mejor recta = buscar el punto más bajo de una superficie.</p>""",
              "El computador no reemplaza el desarrollo matemático: lo verifica y permite analizar más a fondo."))

    D.append((["conc"], "Problema real y datos", f"""
<div class="dos">
 <div>
  <p class="grande"><b>{PROBLEMA}</b></p>
  <ul>
   <li><b>x</b> = estatura [in] (1 in = 2.54 cm) — variable independiente</li>
   <li><b>y</b> = peso promedio [lb] (1 lb = 0.4536 kg) — variable dependiente</li>
   <li>n = {n} observaciones · x̄ = {xm:.0f} in · ȳ = {ym:.4f} lb</li>
   <li>Fuente: {FUENTE}</li>
   <li>Cada y es un <b>promedio</b> de muchas personas, no una persona individual.</li>
  </ul>
 </div>
 <div><table class="tabla mini"><tr><th>i</th><th>xᵢ</th><th>yᵢ</th><th>xᵢ²</th><th>xᵢyᵢ</th><th>yᵢ²</th></tr>{filas}</table></div>
</div>""", "Explicar por qué elegimos este problema: datos reales, verificables, con unidades y con tendencia aproximadamente lineal."))

    D.append((["graf"], "Exploración: diagrama de dispersión", f"""
<div class="dos ancho"><div>{img('01_dispersion.png')}</div>
<div><ul class="grande"><li>Los puntos suben casi en línea recta.</li>
<li>Coeficiente de correlación <b>r = {r_corr:.4f}</b>.</li>
<li>⇒ el modelo <b>y = ax + b</b> es razonable como primera aproximación.</li>
<li>Se nota una leve curvatura (la analizamos después con los residuos).</li></ul></div></div>""",
              "La dispersión justifica el modelo lineal. Mencionar la leve curvatura como adelanto."))

    D.append((["conc"], "Construcción del modelo y función de error", f"""
<div class="formula">ŷᵢ = a·xᵢ + b &nbsp;&nbsp;&nbsp; eᵢ = yᵢ − (a·xᵢ + b)</div>
<div class="formula grande2">S(a, b) = Σ (yᵢ − a·xᵢ − b)²</div>
<p>Con nuestros datos, expandiendo el cuadrado y usando las sumas:</p>
<div class="formula">S(a, b) = 63655a² + 1950ab + 15b² − 268562a − 4102b + 283803</div>
<ul><li>Los errores se elevan al cuadrado para que positivos y negativos no se cancelen y para que S sea derivable.</li>
<li>S es un <b>polinomio de grado 2</b> en a y b ⇒ su gráfica será un paraboloide.</li></ul>""",
              "Recalcar que las variables de S son a y b; los datos son constantes."))

    D.append((["conc", "ind"], "Derivadas parciales", f"""
<p>Regla de la cadena: la derivada de (yᵢ − axᵢ − b)² respecto de a es 2(yᵢ − axᵢ − b)·(−xᵢ).</p>
<div class="formula">∂S/∂a = −2 Σ xᵢ(yᵢ − a·xᵢ − b) = 2(a·Σx² + b·Σx − Σxy) = <b>127310a + 1950b − 268562</b></div>
<div class="formula">∂S/∂b = −2 Σ (yᵢ − a·xᵢ − b) = 2(a·Σx + n·b − Σy) = <b>1950a + 30b − 4102</b></div>
<p class="nota">Comprobación: derivar directamente el polinomio de la diapositiva anterior da lo mismo.</p>""",
              "Cada integrante debe poder hacer esta derivada en el tablero."))

    pasos = [
        "Condición de punto crítico: ∇S = (0, 0)",
        "Ecuaciones normales (dividiendo entre 2):<br>63655a + 975b = 134281<br>975a + 15b = 2051",
        "De la 2ª: b = (2051 − 975a)/15. Sustituyendo en la 1ª y multiplicando por 15:<br>954825a + 1999725 − 950625a = 2014215",
        "4200a = 14490 &nbsp;⇒&nbsp; <b>a* = 3.45 lb/in</b>",
        "b* = (2051 − 3.45·975)/15 = −1312.75/15 &nbsp;⇒&nbsp; <b>b* = −87.5167 lb</b> &nbsp;(= ȳ − a*x̄ ✓)",
        f"S<sub>mín</sub> = Σy² − a*Σxy − b*Σy = 283803 − 463269.45 + 179496.68 = <b>{smin:.4f} lb²</b>",
        "Recta óptima: <span class='resaltado'>ŷ = 3.45x − 87.5167</span>",
    ]
    D.append((["conc", "ind"], "Punto crítico: ∇S = 0 y solución del sistema", f"""
<div class="pasos">{''.join(f"<div class='paso'>{p}</div>" for p in pasos)}</div>
<button class="btn" onclick="siguientePaso(this)">Siguiente paso ▸</button>
<p class="nota">Interpretación: a* = 3.45 lb por cada pulgada (≈ 0.62 kg/cm). b* no tiene sentido físico
(estatura 0): no se extrapola fuera de 58–72 in.</p>""",
              "Revelar los pasos uno por uno mientras se explica. Advertir el redondeo: usar −1312.75/15 exacto para S."))

    D.append((["conc", "ind"], "Hessiano: ¿es realmente un mínimo?", f"""
<div class="dos"><div>
<div class="formula">H = | 2Σx² &nbsp; 2Σx | = | {H[0,0]:.0f} &nbsp; {H[0,1]:.0f} |<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| 2Σx &nbsp;&nbsp; 2n &nbsp;| &nbsp;&nbsp;| {H[1,0]:.0f} &nbsp;&nbsp;&nbsp;&nbsp; {H[1,1]:.0f} |</div>
<ol class="grande"><li>Sₐₐ = {H[0,0]:.0f} &gt; 0</li>
<li>det H = 127310·30 − 1950² = <b>{hv['det']:.0f} &gt; 0</b></li></ol>
<p class="grande">⇒ H definida positiva ⇒ (a*, b*) es <b>mínimo</b>.<br>H es constante ⇒ el mínimo es <b>global y único</b>.</p>
</div><div>
<h3>Verificación numérica</h3>
<ul><li>Autovalores: λ₁ = {lam[0]:.4f}, λ₂ = {lam[1]:.2f} (ambos &gt; 0)</li>
<li>H por diferencias finitas = fórmula (error {hv['error_rel']:.1e})</li>
<li>κ = λ₂/λ₁ = {hv['condicion']:.2e} ⇒ curvas de nivel muy alargadas</li>
<li>det H = 4[nΣx² − (Σx)²] = 4nΣ(xᵢ − x̄)² &gt; 0 siempre que no todos los x sean iguales.</li></ul>
</div></div>""", "Explicar por qué ∇S = 0 no basta: podría ser máximo o silla. El Hessiano lo decide."))

    D.append((["graf", "conc"], "Interpretación geométrica: superficie z = S(a, b)", f"""
<div class="dos ancho"><div class="plot">{plot_div(fig_sup, 560)}</div>
<div><ul><li><b>Paraboloide</b> con un único mínimo (rombo rojo).</li>
<li><b>Trazas</b> (naranja/azul): parábolas que abren hacia arriba.</li>
<li><b>Valle diagonal</b> (línea blanca): b = ȳ − a·x̄ — a y b están acoplados.</li>
<li>Curvas de nivel proyectadas en el piso.</li>
<li><i>Gira con el mouse.</i></li></ul></div></div>""",
              "Girar la superficie en vivo. Explicar que no puede ser ondulada: S es cuadrática."))

    D.append((["graf"], "Trazas y secciones transversales", f"""
{img('04_trazas.png', '50vh')}
<ul><li>b = cte ⇒ parábola en a (curvatura {H[0,0]:.0f}) · a = cte ⇒ parábola en b (curvatura {H[1,1]:.0f}).</li>
<li>El mejor a depende de b (término cruzado 1950ab). En las direcciones de los autovectores la superficie es ~1000 veces más empinada en una que en otra.</li></ul>""",
              "Todas las trazas abren hacia arriba: eso descarta un punto de silla."))

    D.append((["graf"], "Curvas de nivel", f"""
<div class="dos ancho"><div>{img('05_curvas_nivel.png', '52vh')}</div>
<div>{img('13_3d_curvas_nivel.png', '40vh')}
<ul><li>Elipses concéntricas centradas exactamente en (a*, b*).</li>
<li>Muy alargadas en (a, b) por κ ≈ 10⁶; con x centrada son casi círculos.</li></ul></div></div>""",
              "El punto calculado a mano está en el centro de todas las elipses: región de mínimo error."))

    D.append((["graf", "conc"], "Explorador: mueve la recta y mira su altura en S", """
<div class="dos ancho">
 <div id="exp-recta" class="plotbox"></div>
 <div id="exp-niveles" class="plotbox"></div>
</div>
<div class="controles">
 <label>pendiente a: <input type="range" id="sl-a" step="0.005"> <span id="v-a"></span></label>
 <label>altura en x̄ (c = b + a·x̄): <input type="range" id="sl-c" step="0.05"> <span id="v-c"></span></label>
 <button class="btn" onclick="explorarOptimo()">Ir al óptimo</button>
 <div id="v-s" class="resultado"></div>
</div>""", "Mover los deslizadores: la recta se aleja de los datos y el punto sube por las curvas de nivel. 'Ir al óptimo' lleva al fondo."))

    D.append((["graf"], "El paraboloide con x centrada", f"""
<div class="dos ancho"><div class="plot">{plot_div(fig_taz, 520)}</div>
<div><p>Con y = a(x − x̄) + c el término cruzado desaparece (Σ(xᵢ − x̄) = 0):</p>
<div class="formula">H = diag({2*np.sum(u**2):.0f}, {2*n})</div>
<p>Se ve el “tazón” con un solo mínimo. Es el mismo problema; c* = ȳ y b* = c* − a*x̄ = {f4(b)}.</p></div></div>""",
              "Este cambio de variables explica la forma alargada de las curvas de nivel."))

    D.append((["prog"], "Programación: pseudocódigo y diagrama de flujo", """
<div class="dos"><pre class="codigo">ALGORITMO MinimosCuadrados
1. LEER (x_i, y_i) desde datos.csv
2. SI x constante → ERROR
3. Sx, Sy, Sxy, Sxx ← sumas
4. D  ← n·Sxx − Sx²
5. a* ← (n·Sxy − Sx·Sy) / D
6. b* ← (Sy − a*·Sx) / n
7. e_i ← y_i − (a*·x_i + b*)
   S_min ← Σ e_i²
8. H ← [[2Sxx, 2Sx], [2Sx, 2n]]
9. SI H11 > 0 y det H > 0
      → mínimo global
10. RMSE, R²
11. COMPARAR con polyfit/linregress
12. GRAFICAR</pre>
<div class="flujo">
 <div class="nodo ini">Inicio</div><div class="fl">↓</div>
 <div class="nodo">Leer datos</div><div class="fl">↓</div>
 <div class="nodo rombo">¿x no constante?</div><div class="fl">↓ sí</div>
 <div class="nodo">Sumas Σx, Σy, Σxy, Σx²</div><div class="fl">↓</div>
 <div class="nodo">∇S = 0 → a*, b*</div><div class="fl">↓</div>
 <div class="nodo">Hessiano: Sₐₐ &gt; 0, det H &gt; 0</div><div class="fl">↓</div>
 <div class="nodo">Comparar · residuos · gráficas</div><div class="fl">↓</div>
 <div class="nodo ini">Fin</div>
</div></div>""", "Relación código–modelo: cada línea corresponde a una fórmula deducida a mano."))

    D.append((["prog", "conc"], "Comparación: manual vs código propio vs biblioteca", f"""
<table class="tabla"><tr><th>Parámetro</th><th>Analítico (a mano)</th><th>Código propio</th><th>Biblioteca (scipy)</th><th>Δ máx</th></tr>{filas_comp}</table>
<ul><li>Manual vs código: ~10⁻⁵ — solo por <b>redondeo</b> a 4 decimales.</li>
<li>Código vs biblioteca: ~10⁻¹⁴ — error de punto flotante.</li>
<li>Los tres resuelven el mismo sistema ∇S = 0. La biblioteca se usó solo para <b>verificar</b>.</li></ul>""",
              "Aclarar que no usamos una función de regresión incorporada para el ajuste principal."))

    D.append((["graf"], "Recta de ajuste vs otras rectas", f"""
<div class="dos ancho"><div>{img('02_regresion.png', '58vh')}</div>
<div><ul><li>ŷ = 3.45x − 87.52 sigue la tendencia: todos los puntos a menos de 3.2 lb.</li>
<li>Otras rectas tienen S entre 87 y 165 lb² (3 a 5.5 veces más).</li>
<li>Mayores desviaciones: x = 72 (+3.12), x = 58 (+2.42), x = 65 (−1.73).</li></ul>
{img('15_3d_recta_y_superficie.png', '24vh')}</div></div>""",
              "Ninguna recta tiene menor error: lo garantiza el Hessiano."))

    D.append((["graf"], "Análisis de residuos", f"""
<div class="dos ancho"><div>{img('06_residuos_vs_x.png', '36vh')}{img('07_histograma_residuos.png', '28vh')}</div>
<div><ul>
<li><b>Centrados en 0:</b> Σeᵢ = 0 exacto (es ∂S/∂b = 0).</li>
<li><b>Patrón en U:</b> positivos en los extremos, negativos en el centro (3 rachas, corr = {res['corr_curvatura']:.2f}).</li>
<li><b>Dispersión:</b> sin cambio sistemático; el mayor residuo está en x = 72.</li>
<li><b>Atípicos:</b> x = 72 (e/s = 2.04), apenas en el límite.</li>
<li><b>Desviación sistemática:</b> sí — la recta sobreestima en el centro y subestima en los extremos.</li>
<li>Histograma centrado pero asimétrico.</li></ul></div></div>""",
              "Este es el análisis más importante para la reflexión crítica: R² alto no significa modelo perfecto."))

    D.append((["graf", "concl"], "Calidad del ajuste", f"""
<div class="kpis"><div><span>R²</span><b>{cal['R2']:.4f}</b><small>explica el {100*cal['R2']:.1f} % de la variación</small></div>
<div><span>RMSE</span><b>{cal['RMSE']:.2f} lb</b><small>≈ {cal['RMSE']*0.4536:.2f} kg · {100*cal['rmse_rel_media']:.1f} % del peso medio</small></div>
<div><span>S mín</span><b>{smin:.2f} lb²</b><small>suma de errores al cuadrado</small></div></div>
<ul class="grande"><li>R² es muy alto porque los datos son <b>promedios</b> (el ruido individual ya se canceló).</li>
<li>Para estimar el peso promedio entre 58 y 72 in, ±1.4 lb es pequeño.</li>
<li>R² no detecta la curvatura: por eso se analizan los residuos.</li></ul>""",
              "No basta decir que R² es bueno: hay que relacionarlo con los datos y con el propósito."))

    D.append((["graf", "concl"], "Análisis adicional: influencia (B) y modelo cuadrático (C)", f"""
<div class="dos ancho"><div>
<h3>B · Quitar un dato</h3>
<table class="tabla mini"><tr><th>x quitado</th><th>a</th><th>Δa</th><th>b</th><th>Δb</th></tr>{filas_inf}</table>
<p>Más influyente: <b>x = 72</b> (la pendiente baja 3 %). Los extremos pesan más.</p>
<p><b>A · Sensibilidad:</b> +1 lb en un dato cambia a como máximo {np.max(np.abs(sen['da'])):.3f}.</p>
</div><div>
<h3>C · Lineal vs cuadrático</h3>
{img('11_lineal_vs_cuadratico.png', '26vh')}
<table class="tabla mini"><tr><th></th><th>Lineal</th><th>Cuadrático</th></tr>
<tr><td>S</td><td>{cq['S_lineal']:.2f}</td><td>{cq['S_cuad']:.2f}</td></tr>
<tr><td>R²</td><td>{cq['R2_lineal']:.4f}</td><td>{cq['R2_cuad']:.4f}</td></tr>
<tr><td>RMSE</td><td>{cq['RMSE_lineal']:.2f}</td><td>{cq['RMSE_cuad']:.2f}</td></tr></table>
<p>La curvatura es real (F = {cq['F']:.0f}), pero las predicciones difieren ≤ {cq['max_dif_pred']:.1f} lb.</p>
</div></div>""", "Hicimos tres de las cuatro opciones (A, B y C) aunque solo se pedía una."))

    D.append((["prog", "graf"], "Métodos numéricos y forma de la superficie", f"""
<div class="dos ancho"><div class="plot">{plot_div(fig_met, 520)}</div>
<div><table class="tabla mini"><tr><th>Método</th><th>Iteraciones</th><th>¿Converge?</th></tr>
<tr><td>Gradiente simple</td><td>{gd_s['iteraciones']}</td><td>No</td></tr>
<tr><td>Gradiente con x centrada</td><td>{gd_c['iteraciones']}</td><td>Sí</td></tr>
<tr><td>Newton (usa H)</td><td>1</td><td>Sí, exacto</td></tr></table>
<p>El valle estrecho (κ ≈ 10⁶) frena al gradiente simple. Newton usa la curvatura y, como S es cuadrática, llega en un paso.
La solución analítica ∇S = 0 no necesita iterar.</p></div></div>""",
              "Este análisis muestra que el Hessiano no solo dice SI hay mínimo, sino CÓMO es la superficie."))

    D.append((["concl"], "Conclusiones: respuesta a la pregunta central", """
<ol class="grande">
<li>El Cálculo Multivariable convierte “la mejor recta” en <b>el mínimo de S(a, b)</b>.</li>
<li><b>∇S = 0</b> da el sistema 2×2 → a* = 3.45 lb/in, b* = −87.5167 lb, S<sub>mín</sub> = 30.23 lb².</li>
<li>El <b>Hessiano</b> (definido positivo y constante) garantiza un mínimo <b>global y único</b>.</li>
<li>La <b>geometría</b> lo hace visible: paraboloide, trazas parabólicas, curvas de nivel elípticas centradas en (a*, b*).</li>
<li>El <b>computador</b> verificó el resultado (manual = código = biblioteca) y permitió analizar residuos, calidad e influencia.</li>
<li>Por cada pulgada de estatura, el peso promedio aumenta <b>3.45 lb</b> (≈ 0.62 kg/cm).</li>
</ol>""", "Responder explícitamente la pregunta central con las tres perspectivas."))

    D.append((["concl"], "Reflexión crítica: limitaciones del modelo", f"""
<ul class="grande">
<li><b>R² alto ≠ modelo correcto:</b> los residuos en U muestran una relación levemente curva.</li>
<li>El modelo cuadrático reduce S de {cq['S_lineal']:.1f} a {cq['S_cuad']:.1f}, pero cambia las predicciones ≤ {cq['max_dif_pred']:.1f} lb:
la recta sirve para <b>interpretar</b>; la cuadrática para <b>describir con precisión</b>.</li>
<li>Los datos son <b>promedios de 1975</b> de una población específica: no describen a una persona concreta ni a poblaciones actuales.</li>
<li><b>No extrapolar:</b> b* = −87.5 lb muestra que la recta no tiene sentido lejos de 58–72 in.</li>
<li>El ajuste depende más de los extremos (x = 58 y x = 72) que del centro.</li>
<li>El computador no reemplaza el desarrollo: lo verifica, lo visualiza y revela sus límites.</li>
</ul>""", "La reflexión crítica es parte del 15 % de conclusiones."))

    D.append((["ind", "conc"], "Práctica de preguntas (evaluación individual)", f"""
<p>Cada integrante debe responder sin leer. Haz clic en una pregunta para ver la respuesta.
<button class="btn" onclick="preguntaAleatoria()">🎲 Pregunta aleatoria</button>
<button class="btn sec" onclick="cerrarPreguntas()">Cerrar todas</button></p>
<div class="preguntas">{html_preg}</div>""", "Usar esta diapositiva para ensayar antes de la sustentación."))

    # ── HTML ──────────────────────────────────────────────────────────────
    secciones, indice = [], []
    for i, (crit, titulo, cuerpo, notas) in enumerate(D):
        chips = "".join(f"<span class='chip' style='background:{CRITERIOS[c][2]}'>{CRITERIOS[c][0]} {CRITERIOS[c][1]}</span>"
                        for c in crit)
        secciones.append(f"""<section class="slide" data-i="{i}">
<header><h2>{titulo}</h2><div class="chips">{chips}</div></header>
<div class="cuerpo">{cuerpo}</div>
<aside class="notas"><b>Notas para el expositor:</b> {notas}</aside>
</section>""")
        indice.append(f"<li><a href='#' data-ir='{i}'>{i+1}. {titulo}</a></li>")

    js_plotly = plotly.offline.get_plotlyjs()
    html = PLANTILLA.replace("{{PLOTLY}}", js_plotly) \
        .replace("{{SLIDES}}", "\n".join(secciones)) \
        .replace("{{INDICE}}", "\n".join(indice)) \
        .replace("{{EXPLORADOR}}", json.dumps(explorador)) \
        .replace("{{TOTAL}}", str(len(D)))
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    with open(SALIDA, "w", encoding="utf-8") as fh:
        fh.write(html)
    return SALIDA


PLANTILLA = r"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sustentación — Regresión lineal y optimización (Cálculo Multivariable)</title>
<script>{{PLOTLY}}</script>
<style>
:root{--azul:#0B5FCC;--osc:#083A80;--fondo:#f4f7fb;--gris:#5A6472}
*{box-sizing:border-box}
html,body{margin:0;height:100%;font-family:"Segoe UI",Arial,sans-serif;background:var(--fondo);color:#1a2733}
.slide{display:none;position:absolute;inset:0 0 44px 0;padding:22px 46px;overflow:auto}
.slide.activa{display:block;animation:entra .35s ease}
@keyframes entra{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;border-bottom:3px solid #c9d6e8;margin-bottom:14px}
h2{color:var(--osc);margin:4px 0 10px;font-size:30px}
h3{color:var(--azul);margin:6px 0}
.chips{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
.chip{color:white;border-radius:14px;padding:4px 11px;font-size:13px;white-space:nowrap}
.cuerpo{font-size:18px;line-height:1.45}
.grande{font-size:21px} .nota{color:var(--gris);font-size:15px}
.dos{display:grid;grid-template-columns:1fr 1fr;gap:28px;align-items:start}
.dos.ancho{grid-template-columns:1.35fr 1fr}
.tres{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-bottom:18px}
.caja{background:white;border-radius:12px;padding:16px;box-shadow:0 2px 6px rgba(0,0,0,.08)}
img{max-width:100%;border-radius:8px;background:white;box-shadow:0 1px 4px rgba(0,0,0,.1);display:block;margin:0 auto 8px}
.formula{background:white;border-left:5px solid var(--azul);padding:12px 18px;margin:10px 0;font-size:21px;font-family:"Cambria Math","Times New Roman",serif;border-radius:6px}
.grande2{font-size:30px;text-align:center}
.tabla{border-collapse:collapse;background:white;width:100%;font-size:16px}
.tabla th{background:var(--azul);color:white;padding:6px 10px}
.tabla td{border:1px solid #dbe4ee;padding:5px 10px;text-align:center}
.tabla.mini{font-size:14px} .tabla .tot td{font-weight:bold;background:#eef5fc}
.tabla .destacada td{background:#fff1e3;font-weight:bold}
.tabla.rub td{font-size:17px} .tabla.rub a{display:inline-block;margin:2px 4px;padding:2px 9px;border-radius:10px;background:#eef5fc;color:var(--osc);text-decoration:none}
.pos{color:#1B8A4C;font-weight:bold} .neg{color:#E8740C;font-weight:bold}
.portada{text-align:center;padding-top:5vh}
.portada h1{font-size:46px;color:var(--osc);margin:18px 0}
.portada .curso{color:var(--gris);font-size:20px}
.pregunta{max-width:900px;margin:28px auto;padding:22px;border:3px solid var(--azul);border-radius:14px;font-size:25px;background:white;font-weight:600}
.integrantes{font-size:22px;margin-top:30px} .ayuda{color:var(--gris);margin-top:40px}
.paso{display:none;font-size:22px;background:white;border-radius:8px;padding:12px 18px;margin:8px 0;border-left:5px solid #1B8A4C}
.paso.visible{display:block;animation:entra .3s ease}
.resaltado{background:#fff1a8;padding:2px 10px;border-radius:6px;font-weight:bold}
.btn{background:var(--azul);color:white;border:none;border-radius:8px;padding:9px 18px;font-size:16px;cursor:pointer;margin:6px 4px}
.btn.sec{background:var(--gris)} .btn:hover{filter:brightness(1.1)}
.kpis{display:flex;gap:18px;margin:10px 0 18px}
.kpis div{flex:1;background:white;border-radius:12px;padding:16px;box-shadow:0 2px 6px rgba(0,0,0,.08);text-align:center}
.kpis span{color:var(--gris)} .kpis b{display:block;font-size:38px;color:var(--osc)} .kpis small{color:var(--gris)}
.codigo{background:#14213d;color:#e6eefc;padding:16px;border-radius:10px;font-size:15px;line-height:1.5}
.flujo{display:flex;flex-direction:column;align-items:center}
.nodo{background:white;border:2px solid var(--azul);border-radius:8px;padding:7px 16px;font-size:15px;min-width:280px;text-align:center}
.nodo.ini{border-radius:22px;background:var(--azul);color:white}
.nodo.rombo{border-color:#E8740C;background:#fff6ec}
.fl{color:var(--gris);font-size:15px;line-height:1.1}
.plotbox{background:white;border-radius:10px;height:440px}
.controles{display:flex;flex-wrap:wrap;gap:20px;align-items:center;margin-top:10px;font-size:17px}
.controles input{width:260px;vertical-align:middle}
.resultado{font-size:22px;font-weight:bold;color:var(--osc)}
.preguntas{columns:2;column-gap:24px}
.preg{break-inside:avoid;background:white;border-radius:8px;margin:0 0 8px;padding:8px 12px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
.preg summary{cursor:pointer;font-size:16px} .preg p{margin:8px 0 2px;color:#1B5E20;font-size:15.5px}
.preg.resaltada{outline:3px solid #E8740C}
.notas{display:none;position:fixed;left:0;right:0;bottom:44px;background:#14213dee;color:white;padding:12px 46px;font-size:16px}
body.con-notas .slide.activa .notas{display:block}
#barra{position:fixed;left:0;right:0;bottom:0;height:44px;background:white;border-top:1px solid #dbe4ee;display:flex;align-items:center;gap:12px;padding:0 16px;font-size:14px;color:var(--gris)}
#progreso{flex:1;height:6px;background:#e3eaf3;border-radius:3px;overflow:hidden}
#progreso div{height:100%;background:var(--azul);width:0;transition:width .3s}
#barra button{background:none;border:1px solid #c9d6e8;border-radius:6px;padding:4px 10px;cursor:pointer;color:var(--osc)}
#indice{display:none;position:fixed;top:0;bottom:44px;left:0;width:360px;background:white;box-shadow:2px 0 12px rgba(0,0,0,.15);overflow:auto;padding:16px;z-index:10}
#indice.abierto{display:block} #indice ol{padding-left:4px;list-style:none} #indice a{color:var(--osc);text-decoration:none;display:block;padding:4px 6px;border-radius:5px}
#indice a:hover{background:#eef5fc}
@media print{.slide{display:block;position:static;page-break-after:always} #barra,#indice{display:none}}
</style></head><body>
{{SLIDES}}
<nav id="indice"><h3>Índice</h3><ol>{{INDICE}}</ol></nav>
<div id="barra">
 <button onclick="ir(actual-1)">◀</button><button onclick="ir(actual+1)">▶</button>
 <span id="num"></span><div id="progreso"><div></div></div>
 <button onclick="toggleIndice()">☰ Índice (I)</button>
 <button onclick="document.body.classList.toggle('con-notas')">📝 Notas (N)</button>
 <button onclick="pantallaCompleta()">⛶ (F)</button>
</div>
<script>
const slides=[...document.querySelectorAll('.slide')]; const TOTAL={{TOTAL}}; let actual=0;
function ir(i){ if(i<0||i>=slides.length) return;
  slides[actual].classList.remove('activa'); actual=i; slides[i].classList.add('activa');
  document.getElementById('num').textContent=(i+1)+' / '+TOTAL;
  document.querySelector('#progreso div').style.width=(100*(i+1)/TOTAL)+'%';
  history.replaceState(null,'','#'+(i+1));
  setTimeout(()=>slides[i].querySelectorAll('.js-plotly-plot').forEach(p=>Plotly.Plots.resize(p)),60);
  if(slides[i].querySelector('#exp-recta')) iniciarExplorador();
}
function toggleIndice(){document.getElementById('indice').classList.toggle('abierto')}
function pantallaCompleta(){ if(!document.fullscreenElement) document.documentElement.requestFullscreen(); else document.exitFullscreen(); }
document.addEventListener('keydown',ev=>{
  if(ev.target.tagName==='INPUT') return;
  if(['ArrowRight','PageDown',' '].includes(ev.key)){ev.preventDefault();ir(actual+1)}
  else if(['ArrowLeft','PageUp'].includes(ev.key)){ev.preventDefault();ir(actual-1)}
  else if(ev.key==='n'||ev.key==='N') document.body.classList.toggle('con-notas');
  else if(ev.key==='i'||ev.key==='I') toggleIndice();
  else if(ev.key==='f'||ev.key==='F') pantallaCompleta();
  else if(ev.key==='Home') ir(0); else if(ev.key==='End') ir(slides.length-1);
});
document.querySelectorAll('[data-ir]').forEach(a=>a.addEventListener('click',ev=>{ev.preventDefault();
  document.getElementById('indice').classList.remove('abierto'); ir(+a.dataset.ir)}));
function siguientePaso(btn){ const p=btn.parentElement.querySelector('.paso:not(.visible)');
  if(p) p.classList.add('visible'); if(!btn.parentElement.querySelector('.paso:not(.visible)')) btn.textContent='✓ Completo'; }
function preguntaAleatoria(){ const ps=[...document.querySelectorAll('.preg')];
  ps.forEach(p=>{p.open=false;p.classList.remove('resaltada')});
  const p=ps[Math.floor(Math.random()*ps.length)]; p.classList.add('resaltada'); p.scrollIntoView({behavior:'smooth',block:'center'}); }
function cerrarPreguntas(){document.querySelectorAll('.preg').forEach(p=>{p.open=false;p.classList.remove('resaltada')})}

/* ── Explorador ── */
const E={{EXPLORADOR}}; let expListo=false;
function S_de(a,c){let s=0;for(let i=0;i<E.x.length;i++){const r=E.y[i]-a*(E.x[i]-E.xm)-c;s+=r*r}return s}
function actualizarExplorador(){
  const a=+document.getElementById('sl-a').value, c=+document.getElementById('sl-c').value, b=c-a*E.xm, S=S_de(a,c);
  const xs=[Math.min(...E.x)-1,Math.max(...E.x)+1];
  const rx=[],ry=[]; E.x.forEach((xi,i)=>{rx.push(xi,xi,null);ry.push(E.y[i],a*xi+b,null)});
  Plotly.restyle('exp-recta',{x:[xs,rx],y:[[a*xs[0]+b,a*xs[1]+b],ry]},[1,2]);
  Plotly.restyle('exp-niveles',{x:[[a]],y:[[c]]},[1]);
  document.getElementById('v-a').textContent=a.toFixed(3)+' lb/in';
  document.getElementById('v-c').textContent=c.toFixed(2)+' lb  (b = '+b.toFixed(2)+')';
  const pct=100*(S/E.smin-1);
  document.getElementById('v-s').innerHTML='S(a, b) = '+S.toFixed(2)+' lb² '+(pct<0.05?'← <span style="color:#D62728">MÍNIMO</span>':'(+'+pct.toFixed(0)+' % sobre el mínimo)');
}
function iniciarExplorador(){
  if(expListo){Plotly.Plots.resize('exp-recta');Plotly.Plots.resize('exp-niveles');return}
  expListo=true;
  const sa=document.getElementById('sl-a'), sc=document.getElementById('sl-c');
  sa.min=E.amin; sa.max=E.amax; sa.value=E.a+0.4; sc.min=E.cmin; sc.max=E.cmax; sc.value=E.c-3;
  const lay={margin:{l:55,r:10,t:36,b:45},paper_bgcolor:'white'};
  Plotly.newPlot('exp-recta',[
    {x:E.x,y:E.y,mode:'markers',marker:{size:9,color:'#083A80'},name:'datos'},
    {x:[],y:[],mode:'lines',line:{color:'#D62728',width:3},name:'recta'},
    {x:[],y:[],mode:'lines',line:{color:'#E8740C',width:1.5},name:'residuos'}],
    Object.assign({title:{text:'Datos y recta y = a·x + b'},xaxis:{title:{text:'Estatura x [in]'}},
      yaxis:{title:{text:'Peso promedio y [lb]'},range:[Math.min(...E.y)-15,Math.max(...E.y)+15]},showlegend:false},lay),{displaylogo:false,responsive:true});
  Plotly.newPlot('exp-niveles',[
    {type:'contour',x:E.ag,y:E.cg,z:E.S,colorscale:'Viridis',reversescale:true,
     contours:{coloring:'heatmap',showlabels:true,start:E.smin*1.25,end:E.smin*40,size:E.smin*3},colorbar:{title:{text:'S [lb²]'}}},
    {x:[],y:[],mode:'markers',marker:{size:14,color:'#D62728',line:{color:'white',width:2}}},
    {x:[E.a],y:[E.c],mode:'markers',marker:{size:10,symbol:'x',color:'black'}}],
    Object.assign({title:{text:'Curvas de nivel de S (x centrada) — punto rojo = recta actual'},xaxis:{title:{text:'a [lb/in]'}},yaxis:{title:{text:'c = b + a·x̄ [lb]'}},showlegend:false},lay),{displaylogo:false,responsive:true});
  sa.oninput=actualizarExplorador; sc.oninput=actualizarExplorador; actualizarExplorador();
}
function explorarOptimo(){document.getElementById('sl-a').value=E.a;document.getElementById('sl-c').value=E.c;actualizarExplorador()}
const ini=parseInt(location.hash.slice(1)); ir(isNaN(ini)?0:Math.min(Math.max(ini-1,0),slides.length-1));
</script></body></html>"""


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("Presentación generada:", construir())
