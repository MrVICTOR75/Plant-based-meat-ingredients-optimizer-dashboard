import numpy as np
from scipy.optimize import minimize

import config as C
import constraints as K
import m3_evaluate as m3

RESPONSES = ["Energy", "Protein", "Fat", "Carb", "Cost"]
_TARGET_COL = {"Energy": "Energy_out (kJ)", "Protein": "Protein_out (g)",
               "Fat": "Fat_out (g)", "Carb": "Carb_out (g)", "Cost": "Cost (Baht)"}

def _corr(A, B, xi):
    diff = A[:, None, :] - B[None, :, :]
    return np.exp(-xi * (diff ** 2).sum(axis=-1))

def fit_gpr(X, y, xi=None):
    xi = C.GP_XI if xi is None else xi
    n = len(X)
    R = _corr(X, X, xi) + C.GP_NUGGET * np.eye(n)
    Rinv = np.linalg.inv(R)
    one = np.ones(n)
    mu = (one @ Rinv @ y) / (one @ Rinv @ one)
    w = Rinv @ (y - mu * one)
    return {"mu": mu, "w": w, "Rinv": Rinv, "X": X, "xi": xi}

def gpr_predict(model, Xstar):
    Xstar = np.atleast_2d(Xstar)
    r = _corr(Xstar, model["X"], model["xi"])
    return model["mu"] + r @ model["w"]

def fit_all_responses(X, R_df, xi=None):
    return {resp: fit_gpr(X, R_df[_TARGET_COL[resp]].to_numpy(float), xi=xi)
            for resp in RESPONSES}

def loocv_r2(X, R_df, xi=None):
    xi = C.GP_XI if xi is None else xi
    n = len(X)
    R = _corr(X, X, xi) + C.GP_NUGGET * np.eye(n)
    Rinv = np.linalg.inv(R)
    F = np.ones((n, 1))
    M = F.T @ Rinv @ F
    Q = Rinv - Rinv @ F @ np.linalg.inv(M) @ F.T @ Rinv
    diagQ = np.clip(np.diag(Q), 1e-12, None)

    r2 = {}
    for resp in RESPONSES:
        y = R_df[_TARGET_COL[resp]].to_numpy(float)
        e = (Q @ y) / diagQ
        press = float(np.sum(e ** 2))
        ssto = float(((y - y.mean()) ** 2).sum())
        r2[resp] = 1.0 - press / ssto if ssto > 1e-12 else np.nan
    return r2

def d_two_sided(y, L, U):
    T = 0.5 * (L + U)
    if y < L or y > U:
        return 0.0
    return (y - L) / (T - L) if y <= T else (U - y) / (U - T)

def d_cost(y, U):
    if y < 0:
        return 1.0
    return 0.0 if y > U else (U - y) / U

def individual_desirabilities(responses, group_name, cost_u=None):
    d = {r: d_two_sided(responses[r], *C.PRODUCT_GROUPS[group_name][r])
         for r in ["Energy", "Protein", "Fat", "Carb"]}
    d["Cost"] = d_cost(responses["Cost"], C.COST_U if cost_u is None else cost_u)
    return d

def overall_D(responses, group_name, cost_u=None):
    d = individual_desirabilities(responses, group_name, cost_u=cost_u)
    vals = np.array(list(d.values()), float)
    if np.any(vals <= 0):
        return 0.0, d
    return float(np.exp(np.mean(np.log(vals)))), d

def optimize_desirability(models, group_name, n_starts=40, seed=None, cost_u=None):
    rng = np.random.default_rng(C.RANDOM_SEED if seed is None else seed)

    def responses_at(x):
        return {resp: float(gpr_predict(models[resp], np.atleast_2d(x))[0])
                for resp in RESPONSES}

    def neg_D(x):
        return -overall_D(responses_at(x), group_name, cost_u=cost_u)[0]

    cons = K.scipy_constraints()
    lows, highs = K.get_bounds()
    bounds = list(zip(lows, highs))
    starts = K.sample_feasible(n_starts, rng=rng)

    best = None
    for x0 in starts:
        res = minimize(neg_D, x0, method="SLSQP", bounds=bounds,
                       constraints=cons, options={"maxiter": 200, "ftol": 1e-10})
        if not res.success or not K.is_feasible(res.x):
            continue
        D, d = overall_D(responses_at(res.x), group_name, cost_u=cost_u)
        if (best is None) or (D > best["D"]):
            best = {"x": res.x, "D": D, "d": d, "responses": responses_at(res.x)}

    if best is None:
        scored = [(overall_D(responses_at(x), group_name, cost_u=cost_u)[0], x) for x in starts]
        D, x = max(scored, key=lambda t: t[0])
        best = {"x": x, "D": D,
                "d": individual_desirabilities(responses_at(x), group_name, cost_u=cost_u),
                "responses": responses_at(x)}
    return best

def format_best(best):
    x = best["x"]
    return {
        "D": best["D"],
        "formulation_%": {**{C.INGREDIENT_NAMES[ing]: round(100 * C.VARIABLE_SHARE * xi, 2)
                             for ing, xi in zip(C.INGREDIENTS, x)},
                          **{k: round(100 * v, 2) for k, v in C.FIXED_COMPONENTS.items()}},
        "responses": {k: round(v, 2) for k, v in best["responses"].items()},
        "individual_d": {k: round(v, 3) for k, v in best["d"].items()},
    }
