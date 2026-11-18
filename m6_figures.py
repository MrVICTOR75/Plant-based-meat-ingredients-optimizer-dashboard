import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import config as C
import m3_evaluate as m3
import m4_gpr_df as m4

INK = "#1d2430"
MUTED = "#5b6575"
GRID = "#d9dee5"
MARK = "#2f5d8a"
CM = 1 / 2.54
DPI = 300

SHORT = {
    "Latin Hypercube (Rejection)": "LHD-R",
    "Latin Hypercube (Transformation)": "LHD-T",
    "Simplex Lattice": "SLD",
    "Simplex Centroid": "SCD",
    "Maximin (Morris-Mitchell)": "Maximin",
    "Maximum Entropy": "MED",
    "Dirichlet": "Dirichlet",
}

RC = {
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "mathtext.fontset": "stix",
}


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    return buf.getvalue()


def design_points_png(results):
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 4, figsize=(16 * CM, 8.6 * CM), sharex=True, sharey=True)
        for k, (meth, ax) in enumerate(zip(C.DESIGN_METHODS, axes.flat)):
            X = results[meth]["design"][C.INGREDIENTS].to_numpy(float)
            P = m3.product_composition(X) * 100
            ax.scatter(P[:, 0], P[:, 2], s=6, color=MARK, linewidths=0)
            ax.set_title(f"({chr(97 + k)}) {SHORT[meth]}", color=INK, loc="left")
            _style(ax)
        axes.flat[-1].axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel(f"{C.INGREDIENT_NAMES['Gel']} (%)")
        for ax in axes.flat[:len(C.DESIGN_METHODS)]:
            ax.set_xlabel(f"{C.INGREDIENT_NAMES['Dough']} (%)")
            ax.xaxis.set_tick_params(labelbottom=True)
        fig.tight_layout(w_pad=0.6, h_pad=1.0)
        return _png(fig)


RESPONSE_LABELS = [("Energy", "Energy (kJ)"), ("Protein", "Protein (g)"), ("Fat", "Fat (g)"),
                   ("Carb", "Carbohydrate (g)"), ("Cost", "Cost (THB)")]


def _label_points(cs, ax, margin=0.07):
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    levels, pts = [], []
    for lv, path in zip(cs.levels, cs.get_paths()):
        v = path.vertices
        if not len(v):
            continue
        u = (v[:, 0] - x0) / (x1 - x0)
        w = (v[:, 1] - y0) / (y1 - y0)
        clear = np.minimum.reduce([u, 1 - u, w, 1 - w])
        i = int(np.argmax(clear))
        if clear[i] >= margin:
            levels.append(lv)
            pts.append(tuple(v[i]))
    return levels, pts


def responses_png(models, bounds, oil_pct=5.0):
    vs = C.VARIABLE_SHARE * 100
    lows, highs = (np.asarray(b) * vs for b in bounds)
    starch = np.linspace(lows[3], highs[3], 131)
    gel = np.linspace(lows[2], highs[2], 161)
    SS, GG = np.meshgrid(starch, gel)
    dough = vs - oil_pct - SS - GG
    Xg = np.column_stack([dough.ravel(), np.full(dough.size, oil_pct), GG.ravel(), SS.ravel()]) / vs
    outside = (dough < lows[0] - 1e-9) | (dough > highs[0] + 1e-9)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 3, figsize=(16 * CM, 10.5 * CM))
        for k, ((key, label), ax) in enumerate(zip(RESPONSE_LABELS, axes.flat)):
            Z = np.ma.masked_where(outside, m4.gpr_predict(models[key], Xg).reshape(SS.shape))
            ax.set_facecolor("#e5e7eb" if outside.any() else "white")
            cf = ax.contourf(SS, GG, Z, levels=12, cmap="Blues")
            lv = MaxNLocator(nbins=6).tick_values(Z.min(), Z.max())
            lv = lv[(lv > Z.min()) & (lv < Z.max())]
            cs = ax.contour(SS, GG, Z, levels=lv, colors=INK, linewidths=0.4)
            ax.set_xlim(starch[0], starch[-1])
            ax.set_ylim(gel[0], gel[-1])
            lab_lv, lab_pts = _label_points(cs, ax)
            ax.clabel(cs, levels=lab_lv, fmt=lambda v: f"{v:g}", fontsize=6, inline=True, manual=lab_pts)
            ax.set_title(f"({chr(97 + k)}) {label}", color=INK, loc="left")
            ax.set_xlabel(f"{C.INGREDIENT_NAMES['Starch']} (%)")
            ax.set_ylabel(f"{C.INGREDIENT_NAMES['Gel']} (%)")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            cb = fig.colorbar(cf, ax=ax, pad=0.03)
            cb.outline.set_linewidth(0.5)
            cb.ax.tick_params(labelsize=6.5, width=0.5)
        axes.flat[-1].axis("off")
        fig.tight_layout(w_pad=0.8, h_pad=1.0)
        return _png(fig)
