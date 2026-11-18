import numpy as np
import pandas as pd
from scipy.optimize import minimize

import config as C
import constraints as K
import m1_nutrition as m1
import m2_design as m2
import m3_evaluate as m3
import m4_gpr_df as m4

SAMPLE_SIZES = [100, 300]
SEEDS = range(1, 31)
COST_BOUNDS = [5.0, 10.0, 30.0]
OUTPUT_XLSX = "Results.xlsx"
PCT = {ing: f"{C.INGREDIENT_NAMES[ing]} (% product)" for ing in C.INGREDIENTS}


def true_D(x, group, matrix, cost_u=None):
    return m4.overall_D(m3.responses_from_x(x, matrix), group, cost_u=cost_u)[0]


def true_optimum(group, matrix, cost_u=None, n_starts=250):
    lows, highs = K.get_bounds()
    best_x, best_D = None, -1.0
    for x0 in K.sample_feasible(n_starts, rng=np.random.default_rng(0)):
        r = minimize(lambda x: -true_D(x, group, matrix, cost_u), x0, method="SLSQP",
                     bounds=list(zip(lows, highs)), constraints=K.scipy_constraints(),
                     options={"maxiter": 300, "ftol": 1e-12})
        if K.is_feasible(r.x) and -r.fun > best_D:
            best_x, best_D = r.x, -r.fun
    return best_x, best_D


def test_rmse(models, matrix, n_test=2000):
    Xt = K.sample_feasible(n_test, rng=np.random.default_rng(2024))
    ev = m3.evaluate_design(m2.to_dataframe(Xt), matrix)
    return {r: float(np.sqrt(np.mean((m4.gpr_predict(models[r], Xt) - ev[m4._TARGET_COL[r]].to_numpy(float)) ** 2)))
            for r in m4.RESPONSES}


def fit_design(method, n, seed, matrix):
    design, info = m2.generate_design(method, n, seed=seed)
    evald = m3.evaluate_design(design, matrix)
    X = design[C.INGREDIENTS].to_numpy(float)
    return design, info, evald, X, m4.fit_all_responses(X, evald)


def run():
    K.reset_bounds()
    matrix = m1.load_matrix()
    groups = list(C.PRODUCT_GROUPS)

    optima = {g: true_optimum(g, matrix) for g in groups}
    opt_rows = []
    for u in COST_BOUNDS:
        for g in groups:
            x, D = optima[g] if u == C.COST_U else true_optimum(g, matrix, cost_u=u)
            resp = m3.responses_from_x(x, matrix)
            row = {"Cost bound U": u, "Group": g, "D*": D}
            row.update({PCT[ing]: 100 * C.VARIABLE_SHARE * v for ing, v in zip(C.INGREDIENTS, x)})
            row.update(resp)
            row.update({f"d {k}": v for k, v in m4.individual_desirabilities(resp, g, cost_u=u).items()})
            opt_rows.append(row)
        print(f"optimum U={u}")

    rows, rmse_rows = [], []
    for n in SAMPLE_SIZES:
        for method in C.DESIGN_METHODS:
            design, info, evald, X, models = fit_design(method, n, C.RANDOM_SEED, matrix)
            r2 = m4.loocv_r2(X, evald)
            rmse_rows.append({"n requested": n, "Method": method, "Points": info["actual_n"],
                              "Min distance": info["min_distance"],
                              "Mean R2": float(np.nanmean(list(r2.values()))),
                              **{f"RMSE {k}": v for k, v in test_rmse(models, matrix).items()}})
            for group in groups:
                best = m4.optimize_desirability(models, group, seed=C.RANDOM_SEED)
                Dt = true_D(best["x"], group, matrix)
                row = {"Group": group, "n requested": n, "Method": method, "Points": info["actual_n"],
                       "D (surrogate)": best["D"], "D (true)": Dt, "Regret": optima[group][1] - Dt}
                row.update({PCT[ing]: 100 * C.VARIABLE_SHARE * v for ing, v in zip(C.INGREDIENTS, best["x"])})
                rows.append(row)
            print(f"n={n:<4d} {method}")

    seed_rows = []
    for n in SAMPLE_SIZES:
        for method in C.DESIGN_METHODS:
            for seed in SEEDS:
                _, _, _, _, models = fit_design(method, n, seed, matrix)
                for group in groups:
                    best = m4.optimize_desirability(models, group, seed=seed)
                    seed_rows.append({"n requested": n, "Method": method, "Seed": seed, "Group": group,
                                      "Regret": optima[group][1] - true_D(best["x"], group, matrix)})
            print(f"seeds n={n:<4d} {method}")

    seeds = pd.DataFrame(seed_rows)
    worst = seeds.groupby(["n requested", "Method", "Seed"]).Regret.max().reset_index()
    summary = worst.groupby(["n requested", "Method"], sort=False).Regret.describe(
        percentiles=[0.25, 0.5, 0.75])[["25%", "50%", "75%", "max"]].reset_index()
    summary.columns = ["n requested", "Method", "Q1", "Median", "Q3", "Max"]

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        pd.DataFrame(rmse_rows).to_excel(w, sheet_name="Designs", index=False)
        pd.DataFrame(rows).to_excel(w, sheet_name="All runs", index=False)
        pd.DataFrame(opt_rows).to_excel(w, sheet_name="True optimum", index=False)
        summary.to_excel(w, sheet_name="Regret 30 seeds", index=False)
        seeds.to_excel(w, sheet_name="Seed runs", index=False)
        matrix.to_excel(w, sheet_name="Design ingredients", index=False)
        m1.load_raw_matrix().reset_index().to_excel(w, sheet_name="Raw materials (USDA)", index=False)


if __name__ == "__main__":
    run()
    print(f"Saved {OUTPUT_XLSX}")
