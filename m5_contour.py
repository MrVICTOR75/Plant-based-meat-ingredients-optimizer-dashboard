import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as C
import m4_gpr_df as m4

INK, MUTED, GRID, MARK = "#1d2430", "#5b6575", "#d9dee5", "#2f5d8a"
FONT = dict(family="Times New Roman, Liberation Serif, serif", size=13, color=INK)
SHORT = {"Latin Hypercube (Rejection)": "LHD-R", "Latin Hypercube (Transformation)": "LHD-T",
         "Simplex Lattice": "SLD", "Simplex Centroid": "SCD", "Maximin (Morris-Mitchell)": "Maximin",
         "Maximum Entropy": "MED", "Dirichlet": "Dirichlet"}
LABELS = {"Energy": "Energy (kJ)", "Protein": "Protein (g)", "Fat": "Fat (g)",
          "Carb": "Carbohydrate (g)", "Cost": "Cost (THB)"}
VS = 100 * C.VARIABLE_SHARE


def _style(fig):
    fig.update_layout(font=FONT, plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                      margin=dict(l=60, r=20, t=40, b=50))
    fig.update_xaxes(showline=True, linecolor=MUTED, ticks="outside", tickcolor=MUTED, gridcolor=GRID,
                     zeroline=False, mirror=False)
    fig.update_yaxes(showline=True, linecolor=MUTED, ticks="outside", tickcolor=MUTED, gridcolor=GRID,
                     zeroline=False, mirror=False)


def _left_titles(fig, titles):
    for k, t in enumerate(titles):
        xa = fig.layout[f"xaxis{'' if k == 0 else k + 1}"]
        ya = fig.layout[f"yaxis{'' if k == 0 else k + 1}"]
        fig.add_annotation(text=t, x=xa.domain[0], y=ya.domain[1], xref="paper", yref="paper",
                           xanchor="left", yanchor="bottom", showarrow=False, font=dict(FONT, size=15))


def design_points_fig(results):
    rows, cols = 2, 4
    fig = make_subplots(rows, cols, horizontal_spacing=0.06, vertical_spacing=0.16)
    titles = []
    for k, meth in enumerate(C.DESIGN_METHODS):
        r, c = divmod(k, cols)
        X = results[meth]["design"][C.INGREDIENTS].to_numpy(float) * VS
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 2], mode="markers",
                                 marker=dict(size=5, color=MARK),
                                 hovertemplate="Meat dough %{x:.1f}%<br>1% agar gel %{y:.1f}%<extra></extra>"),
                      r + 1, c + 1)
        fig.update_xaxes(title_text="Meat dough (%)", row=r + 1, col=c + 1)
        if c == 0:
            fig.update_yaxes(title_text="1% agar gel (%)", row=r + 1, col=c + 1)
        titles.append(f"({chr(97 + k)}) {SHORT[meth]}")
    fig.update_xaxes(visible=False, row=2, col=4)
    fig.update_yaxes(visible=False, row=2, col=4)
    _style(fig)
    _left_titles(fig, titles)
    fig.update_layout(height=620)
    return fig


def _nice_levels(z, n=6):
    lo, hi = np.nanmin(z), np.nanmax(z)
    raw = (hi - lo) / n
    mag = 10 ** np.floor(np.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10 * mag)
    return np.ceil(lo / step) * step, step


def response_contours_fig(models, bounds, oil_pct=5.0):
    lows, highs = (np.asarray(b) * VS for b in bounds)
    s = np.linspace(lows[3], highs[3], 131)
    g = np.linspace(lows[2], highs[2], 161)
    SS, GG = np.meshgrid(s, g)
    dough = VS - oil_pct - SS - GG
    Xg = np.column_stack([dough.ravel(), np.full(dough.size, oil_pct), GG.ravel(), SS.ravel()]) / VS
    outside = (dough < lows[0] - 1e-9) | (dough > highs[0] + 1e-9)

    rows, cols = 2, 3
    fig = make_subplots(rows, cols, horizontal_spacing=0.13, vertical_spacing=0.18)
    titles = []
    for k, resp in enumerate(m4.RESPONSES):
        r, c = divmod(k, cols)
        Z = m4.gpr_predict(models[resp], Xg).reshape(SS.shape)
        Z[outside] = np.nan
        if outside.any():
            fig.add_trace(go.Heatmap(x=s, y=g, z=np.where(outside, 1.0, np.nan), showscale=False,
                                     colorscale=[[0, "#e5e7eb"], [1, "#e5e7eb"]], hoverinfo="skip"),
                          r + 1, c + 1)
        xa = fig.layout[f"xaxis{'' if k == 0 else k + 1}"].domain
        ya = fig.layout[f"yaxis{'' if k == 0 else k + 1}"].domain
        fig.add_trace(go.Contour(
            x=s, y=g, z=Z, colorscale="Blues", ncontours=12, line=dict(width=0),
            contours=dict(coloring="fill", showlines=False),
            colorbar=dict(x=xa[1] + 0.01, y=(ya[0] + ya[1]) / 2, len=ya[1] - ya[0], thickness=12,
                          outlinecolor=MUTED, outlinewidth=0.5, tickfont=dict(size=11, color=MUTED)),
            hovertemplate="Corn starch %{x:.1f}%<br>1% agar gel %{y:.1f}%<br>" + LABELS[resp]
                          + " %{z:.2f}<extra></extra>"), r + 1, c + 1)
        start, step = _nice_levels(Z)
        fig.add_trace(go.Contour(
            x=s, y=g, z=Z, showscale=False, hoverinfo="skip",
            contours=dict(coloring="none", start=start, end=np.nanmax(Z), size=step, showlabels=True,
                          labelfont=dict(size=11, color=INK)),
            line=dict(color=INK, width=0.6)), r + 1, c + 1)
        fig.update_xaxes(title_text="Corn starch (%)", range=[s[0], s[-1]], showgrid=False, row=r + 1, col=c + 1)
        fig.update_yaxes(title_text="1% agar gel (%)", range=[g[0], g[-1]], showgrid=False, row=r + 1, col=c + 1)
        titles.append(f"({chr(97 + k)}) {LABELS[resp]}")
    fig.update_xaxes(visible=False, row=2, col=3)
    fig.update_yaxes(visible=False, row=2, col=3)
    _style(fig)
    _left_titles(fig, titles)
    fig.update_layout(height=680)
    return fig
