"""
modelos_3d.py
-------------
Modelos 3D INTERACTIVOS (Plotly) para la sustentación:

    results/html/modelos_3d.html   (funciona SIN internet: plotly va incrustado)

  Modelo 1 · Superficie z = S(a, b) con el mínimo, las TRAZAS (cortes con los
             planos a = a* y b = b*) y las CURVAS DE NIVEL proyectadas.
  Modelo 2 · La misma superficie con x centrada (c = b + a·x̄): el "tazón".
  Modelo 3 · EXPLORADOR: al mover el deslizador cambia la recta y = ax + b
             sobre los datos y, al mismo tiempo, el punto (a, b, S) se mueve
             sobre la superficie. Muestra que la mejor recta = el fondo del tazón.
  Modelo 4 · Métodos numéricos en 3D: gradiente simple, gradiente con x
             centrada y Newton bajando por la superficie.
"""

import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline

from derivadas import hessiana
from modelo import error_cuadratico
from datos import ETIQUETA_X, ETIQUETA_Y, PROBLEMA, FUENTE

CARPETA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "html")
ESCALA = "Viridis"
ROJO, NARANJA, VERDE, AZUL = "#D62728", "#E8740C", "#1B8A4C", "#0B5FCC"


def _S(x, y, A, B):
    return ((y[None, None, :] - A[..., None] * x[None, None, :] - B[..., None]) ** 2).sum(-1)


def _malla_eliptica(x, y, a_s, b_s, delta, nr=60, nt=121):
    """Malla en la base de autovectores de H que cubre S ≤ S_mín + delta."""
    lam, V = np.linalg.eigh(hessiana(x))
    r = np.linspace(0, 1, nr)[:, None]
    th = np.linspace(0, 2 * np.pi, nt)[None, :]
    s1, s2 = np.sqrt(2 * delta / lam[0]), np.sqrt(2 * delta / lam[1])
    A = a_s + r * (np.cos(th) * s1 * V[0, 0] + np.sin(th) * s2 * V[0, 1])
    B = b_s + r * (np.cos(th) * s1 * V[1, 0] + np.sin(th) * s2 * V[1, 1])
    return A, B, _S(x, y, A, B)


def _escena(xt, yt, zt="S(a, b) [lb²]"):
    return dict(xaxis_title=xt, yaxis_title=yt, zaxis_title=zt,
                camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
                aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.75))


# ═════════════════════ MODELO 1 · SUPERFICIE + TRAZAS + NIVELES ═════════════════════
def modelo_superficie(x, y, a_s, b_s, s_min, da=0.08, db=5.0):
    """
    Rectángulo a* ± da, b* ± db alrededor del mínimo. En él el VALLE diagonal
    (a y b acoplados) y las dos trazas parabólicas se ven con claridad.
    """
    a_v = np.linspace(a_s - da, a_s + da, 90)
    b_v = np.linspace(b_s - db, b_s + db, 90)
    A, B = np.meshgrid(a_v, b_v)
    S = _S(x, y, A, B)
    zt = float(S.max())
    fig = go.Figure()
    fig.add_trace(go.Surface(x=A, y=B, z=S, colorscale=ESCALA, opacity=0.92, name="S(a, b)",
                             colorbar=dict(title="S [lb²]", len=0.7),
                             contours=dict(z=dict(show=True, usecolormap=True, project_z=True,
                                                  start=s_min + 20, end=zt, size=zt / 14)),
                             hovertemplate="a=%{x:.4f}<br>b=%{y:.3f}<br>S=%{z:.2f}<extra></extra>"))
    # traza con el plano b = b*  (parábola en a)
    fig.add_trace(go.Scatter3d(x=a_v, y=np.full_like(a_v, b_s),
                               z=[error_cuadratico(x, y, av, b_s) for av in a_v], mode="lines",
                               line=dict(color=NARANJA, width=9), name="traza con el plano b = b* (parábola en a)"))
    # traza con el plano a = a*  (parábola en b)
    fig.add_trace(go.Scatter3d(x=np.full_like(b_v, a_s), y=b_v,
                               z=[error_cuadratico(x, y, a_s, bv) for bv in b_v], mode="lines",
                               line=dict(color=AZUL, width=9), name="traza con el plano a = a* (parábola en b)"))
    # fondo del valle: para cada a, el b que minimiza S  (b = ȳ − a·x̄)
    fig.add_trace(go.Scatter3d(x=a_v, y=y.mean() - a_v * x.mean(),
                               z=[error_cuadratico(x, y, av, y.mean() - av * x.mean()) for av in a_v],
                               mode="lines", line=dict(color="white", width=5, dash="dash"),
                               name="fondo del valle: b = ȳ − a·x̄"))
    # planos de corte (transparentes)
    fig.add_trace(go.Mesh3d(x=[a_v[0], a_v[-1], a_v[-1], a_v[0]], y=[b_s] * 4, z=[0, 0, zt, zt],
                            i=[0, 0], j=[1, 2], k=[2, 3], color=NARANJA, opacity=0.13,
                            name="plano b = b*", showlegend=True, hoverinfo="skip"))
    fig.add_trace(go.Mesh3d(x=[a_s] * 4, y=[b_v[0], b_v[-1], b_v[-1], b_v[0]], z=[0, 0, zt, zt],
                            i=[0, 0], j=[1, 2], k=[2, 3], color=AZUL, opacity=0.10,
                            name="plano a = a*", showlegend=True, hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(x=[a_s], y=[b_s], z=[s_min], mode="markers+text",
                               marker=dict(size=7, color=ROJO, symbol="diamond"),
                               text=["mínimo"], textposition="top center",
                               name=f"mínimo (a*, b*) = ({a_s:.4f}, {b_s:.4f}), S = {s_min:.4f}"))
    esc = _escena("a (pendiente) [lb/in]", "b (intercepto) [lb]")
    esc["camera"] = dict(eye=dict(x=-1.5, y=-1.7, z=1.0))
    fig.update_layout(scene=esc, height=720, margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", x=0, y=-0.02, bgcolor="rgba(255,255,255,0.8)"))
    return fig


# ═════════════════════ MODELO 2 · TAZÓN CON x CENTRADA ═════════════════════
def modelo_tazon(x, y, a_s, b_s, s_min):
    xm = x.mean(); u = x - xm; c_s = b_s + a_s * xm
    delta = 6 * s_min
    da = np.sqrt(2 * delta / (2 * np.sum(u ** 2))); dc = np.sqrt(2 * delta / (2 * len(x)))
    r = np.linspace(0, 1, 50)[:, None]; th = np.linspace(0, 2 * np.pi, 121)[None, :]
    A = a_s + r * da * np.cos(th); C = c_s + r * dc * np.sin(th)
    S = _S(u, y, A, C)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=A, y=C, z=S, colorscale=ESCALA, opacity=0.95,
                             colorbar=dict(title="S [lb²]", len=0.7),
                             contours=dict(z=dict(show=True, usecolormap=True, project_z=True,
                                                  start=s_min, end=7 * s_min, size=s_min * 0.75)),
                             hovertemplate="a=%{x:.4f}<br>c=%{y:.3f}<br>S=%{z:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=[a_s], y=[c_s], z=[s_min], mode="markers",
                               marker=dict(size=7, color=ROJO, symbol="diamond"),
                               name=f"mínimo: a* = {a_s:.4f}, c* = ȳ = {c_s:.4f}"))
    fig.update_layout(scene=_escena("a [lb/in]", "c = b + a·x̄ [lb]"), height=600,
                      margin=dict(l=0, r=0, t=10, b=0), legend=dict(x=0.01, y=0.99))
    return fig


# ═════════════════════ MODELO 3 · EXPLORADOR RECTA ↔ PUNTO ═════════════════════
def modelo_explorador(x, y, a_s, b_s, s_min):
    """
    Recorrido en variables centradas (rotar la recta alrededor de (x̄, ȳ) y luego
    desplazarla verticalmente). Cada cuadro muestra la recta sobre los datos y el
    punto correspondiente sobre la superficie S.
    """
    xm = x.mean(); u = x - xm; c_s = b_s + a_s * xm
    delta = 6 * s_min
    da = np.sqrt(2 * delta / (2 * np.sum(u ** 2))); dc = np.sqrt(2 * delta / (2 * len(x)))
    r = np.linspace(0, 1, 40)[:, None]; th = np.linspace(0, 2 * np.pi, 91)[None, :]
    A = a_s + r * da * np.cos(th); C = c_s + r * dc * np.sin(th)
    S = _S(u, y, A, C)

    t = np.linspace(-1, 1, 9)
    camino = ([(a_s + 0.95 * da * ti, c_s) for ti in t] +           # gira la recta
              [(a_s, c_s + 0.95 * dc * ti) for ti in t] +           # la sube y baja
              [(a_s, c_s)])                                          # óptima
    xs = np.array([x.min() - 1, x.max() + 1])

    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55],
                        specs=[[{"type": "xy"}, {"type": "scene"}]],
                        subplot_titles=("Datos y recta y = a·x + b", "Punto sobre la superficie S (x centrada: c = b + a·x̄)"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(size=9, color="#083A80"),
                             name="datos"), 1, 1)

    def recta(a, c):
        b = c - a * xm
        return go.Scatter(x=xs, y=a * xs + b, mode="lines", line=dict(color=ROJO, width=3),
                          name="recta actual")

    def residuos(a, c):
        b = c - a * xm
        xx, yy = [], []
        for xi, yi in zip(x, y):
            xx += [xi, xi, None]; yy += [yi, a * xi + b, None]
        return go.Scatter(x=xx, y=yy, mode="lines", line=dict(color=NARANJA, width=1.5),
                          name="residuos eᵢ")

    def punto(a, c):
        return go.Scatter3d(x=[a], y=[c], z=[error_cuadratico(u, y, a, c)], mode="markers",
                            marker=dict(size=9, color=ROJO, line=dict(color="white", width=2)),
                            name="(a, c, S)")

    a0, c0 = camino[0]
    fig.add_trace(recta(a0, c0), 1, 1)
    fig.add_trace(residuos(a0, c0), 1, 1)
    # superficie en variables centradas (a, c): el tazón (c = b + a·x̄)
    fig.add_trace(go.Surface(x=A, y=C, z=S, colorscale=ESCALA, opacity=0.8,
                             showscale=False, hoverinfo="skip"), 1, 2)
    fig.add_trace(go.Scatter3d(x=[a_s], y=[c_s], z=[s_min], mode="markers",
                               marker=dict(size=5, color="white", line=dict(color="black", width=1)),
                               name="mínimo (a*, b*)"), 1, 2)
    fig.add_trace(punto(a0, c0), 1, 2)

    frames, pasos = [], []
    for k, (a, c) in enumerate(camino):
        Sk = error_cuadratico(u, y, a, c)
        nombre = f"f{k}"
        frames.append(go.Frame(name=nombre, data=[recta(a, c), residuos(a, c), punto(a, c)],
                               traces=[1, 2, 5],
                               layout=dict(annotations=[
                                   fig.layout.annotations[0], fig.layout.annotations[1],
                                   dict(text=f"a = {a:.4f} lb/in · b = {c - a * xm:.3f} lb · "
                                             f"<b>S = {Sk:.2f} lb²</b>"
                                             + ("  ← MÍNIMO" if k == len(camino) - 1 else ""),
                                        x=0.5, y=-0.19, xref="paper", yref="paper",
                                        showarrow=False, font=dict(size=15))])))
        etiqueta = ("óptima" if k == len(camino) - 1
                    else ("girar " if k < 9 else "subir ") + f"{(k % 9) - 4:+d}")
        pasos.append(dict(method="animate", label=etiqueta,
                          args=[[nombre], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                               transition=dict(duration=0))]))
    fig.frames = frames
    fig.add_annotation(text=f"S = {error_cuadratico(u, y, a0, c0):.2f} lb²", x=0.5, y=-0.19,
                       xref="paper", yref="paper", showarrow=False, font=dict(size=15))
    fig.update_layout(
        height=660, margin=dict(l=40, r=0, t=40, b=140),
        xaxis=dict(title=ETIQUETA_X), yaxis=dict(title=ETIQUETA_Y, range=[y.min() - 12, y.max() + 12]),
        scene=_escena("a [lb/in]", "c = b + a·x̄ [lb]"),
        sliders=[dict(active=0, steps=pasos, y=-0.06, len=0.9, x=0.05,
                      currentvalue=dict(prefix="paso: "))],
        updatemenus=[dict(type="buttons", x=0.0, y=1.12, direction="left", buttons=[
            dict(label="▶ Recorrer", method="animate",
                 args=[None, dict(frame=dict(duration=700, redraw=True), fromcurrent=True)]),
            dict(label="⏸", method="animate",
                 args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
        showlegend=False)
    return fig


# ═════════════════════ MODELO 4 · MÉTODOS NUMÉRICOS EN 3D ═════════════════════
def modelo_metodos(x, y, a_s, b_s, gd_simple, gd_centrado, newton):
    a_v = np.linspace(-0.3, 4.2, 90); b_v = np.linspace(-220, 15, 90)
    A, B = np.meshgrid(a_v, b_v)
    S = _S(x, y, A, B)
    Z = np.log10(S)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=A, y=B, z=Z, colorscale=ESCALA, opacity=0.85,
                             colorbar=dict(title="log₁₀ S", len=0.7),
                             hovertemplate="a=%{x:.3f}<br>b=%{y:.2f}<br>log S=%{z:.2f}<extra></extra>"))
    for res, col, nom in [(gd_simple, NARANJA, "gradiente simple (no converge)"),
                          (gd_centrado, VERDE, "gradiente con x centrada")]:
        tr = res["trayectoria"]
        pa = np.array([p[0] for p in tr]); pb = np.array([p[1] for p in tr])
        pz = np.log10(np.array([p[2] for p in tr])) + 0.03
        fig.add_trace(go.Scatter3d(x=pa, y=pb, z=pz, mode="lines+markers",
                                   line=dict(color=col, width=6), marker=dict(size=2, color=col),
                                   name=f"{nom}: {res['iteraciones']} iter."))
    S0 = error_cuadratico(x, y, 0, 0)
    fig.add_trace(go.Scatter3d(x=[0, newton["a_star"]], y=[0, newton["b_star"]],
                               z=[np.log10(S0) + 0.03, np.log10(error_cuadratico(x, y, a_s, b_s)) + 0.03],
                               mode="lines", line=dict(color=ROJO, width=6, dash="dash"),
                               name="Newton: 1 paso"))
    fig.add_trace(go.Scatter3d(x=[a_s], y=[b_s], z=[np.log10(error_cuadratico(x, y, a_s, b_s))],
                               mode="markers", marker=dict(size=7, color=ROJO, symbol="diamond"),
                               name="mínimo (a*, b*)"))
    fig.update_layout(scene=_escena("a [lb/in]", "b [lb]", "log₁₀ S(a, b)"), height=640,
                      margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"))
    return fig


# ═════════════════════ PÁGINA ═════════════════════
def _div(fig):
    return plotly.offline.plot(fig, include_plotlyjs=False, output_type="div",
                               config={"displaylogo": False, "responsive": True},
                               auto_play=False)


def generar_modelos_3d(x, y, a_s, b_s, gd_simple, gd_centrado, newton, hv, calidad):
    os.makedirs(CARPETA, exist_ok=True)
    s_min = error_cuadratico(x, y, a_s, b_s)
    lam = hv["autovalores"]
    secciones = [
        ("1 · Superficie de error z = S(a, b), trazas y curvas de nivel",
         f"""<p>Cada punto (a, b) del piso es una recta posible; su altura es el error total S(a, b).
         La superficie es un <b>paraboloide elíptico</b> (S es un polinomio de grado 2 con Hessiano
         constante y definido positivo), así que tiene <b>un único mínimo</b>: el rombo rojo
         (a*, b*) = ({a_s:.4f}, {b_s:.4f}), S<sub>mín</sub> = {s_min:.4f} lb².</p>
         <ul><li><b>Trazas</b> (naranja y azul): al cortar con los planos b = b* y a = a* se obtienen
         parábolas que abren hacia arriba, con curvaturas S<sub>aa</sub> = {hv['H'][0,0]:.0f} y
         S<sub>bb</sub> = {hv['H'][1,1]:.0f}.</li>
         <li><b>Curvas de nivel</b> (proyectadas en el piso): elipses
         concéntricas centradas en (a*, b*), muy alargadas porque κ = λ₂/λ₁ = {hv['condicion']:.2e}.</li>
         <li><b>Valle diagonal</b> (línea blanca): para cada a, el mejor b es b = ȳ − a·x̄. Por eso a y b
         están acoplados: si sube la pendiente, el intercepto debe bajar 65 lb por cada lb/in.</li>
         <li>Se muestra la región a* ± 0.08, b* ± 5 alrededor del mínimo. Gira con el mouse, acerca con la
         rueda y pasa el cursor para ver los valores.</li></ul>""",
         modelo_superficie(x, y, a_s, b_s, s_min)),
        ("2 · La misma función con x centrada: el “tazón”",
         f"""<p>Con y = a(x − x̄) + c, donde c = b + a·x̄, el término cruzado desaparece (Σ(xᵢ − x̄) = 0) y el
         Hessiano queda diagonal, H = diag({2*np.sum((x-x.mean())**2):.0f}, {2*len(x)}). La superficie se ve
         como un tazón casi redondo. Es el mismo problema con los ejes cambiados: el mínimo da
         c* = ȳ = {y.mean():.4f}, de donde b* = c* − a*·x̄ = {b_s:.4f}.</p>""",
         modelo_tazon(x, y, a_s, b_s, s_min)),
        ("3 · Explorador: la mejor recta es el fondo de la superficie",
         """<p>Mueve el deslizador o presiona <b>▶ Recorrer</b>. Primero la recta <b>gira</b> alrededor del centro
         de los datos (x̄, ȳ) y luego <b>sube y baja</b>. A la izquierda se ven la recta y sus residuos; a la
         derecha, el punto (a, b, S) correspondiente sobre la superficie. Cuanto más lejos queda la
         recta de los datos, más alto está el punto. La recta óptima (último paso) es exactamente el
         fondo del paraboloide.</p>""",
         modelo_explorador(x, y, a_s, b_s, s_min)),
        ("4 · Métodos numéricos bajando por la superficie (escala log)",
         f"""<p>Desde (0, 0): el <b>gradiente simple</b> cae rápido al valle y luego avanza casi nada
         ({gd_simple['iteraciones']} iteraciones sin converger, porque κ ≈ {hv['condicion']:.0e}). Con
         <b>x centrada</b> converge en {gd_centrado['iteraciones']} iteraciones. <b>Newton</b>, que usa
         el Hessiano, llega en <b>un paso</b>, porque la aproximación de Taylor de orden 2 de una función
         cuadrática es exacta. La solución analítica ∇S = 0 no necesita iterar.</p>""",
         modelo_metodos(x, y, a_s, b_s, gd_simple, gd_centrado, newton)),
    ]

    js = plotly.offline.get_plotlyjs()
    cuerpo = "\n".join(
        f'<section><h2>{t}</h2><div class="txt">{txt}</div>{_div(fig)}</section>'
        for t, txt, fig in secciones)
    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Modelos 3D — Regresión lineal y optimización</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:1250px;margin:0 auto;padding:24px;color:#1a2733;background:#f7f9fc}}
h1{{color:#083A80;margin-bottom:4px}} h2{{color:#0B5FCC;border-bottom:2px solid #c9d6e8;padding-bottom:4px;margin-top:40px}}
section{{background:white;border-radius:12px;padding:8px 22px 18px;margin:18px 0;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.txt{{font-size:15.5px;line-height:1.5}} .res{{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0}}
.res div{{background:white;border-radius:10px;padding:10px 16px;box-shadow:0 1px 3px rgba(0,0,0,.08)}}
.res b{{display:block;font-size:22px;color:#083A80}} small{{color:#5a6472}}
</style>
<script>{js}</script></head><body>
<h1>Modelos 3D interactivos — Regresión lineal por mínimos cuadrados</h1>
<p><b>Problema:</b> {PROBLEMA}. <small>Fuente: {FUENTE}.</small></p>
<div class="res">
 <div>recta óptima<b>ŷ = {a_s:.4f}x {b_s:+.4f}</b></div>
 <div>S mínimo<b>{s_min:.4f} lb²</b></div>
 <div>det H<b>{hv['det']:.0f} &gt; 0</b></div>
 <div>autovalores de H<b>{lam[0]:.4f} · {lam[1]:.0f}</b></div>
 <div>R² · RMSE<b>{calidad['R2']:.4f} · {calidad['RMSE']:.3f} lb</b></div>
</div>
{cuerpo}
<p><small>Generado por src/main.py → src/modelos_3d.py. Plotly va incluido en el archivo: no necesita internet.</small></p>
</body></html>"""
    ruta = os.path.join(CARPETA, "modelos_3d.html")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(html)
    return ruta
