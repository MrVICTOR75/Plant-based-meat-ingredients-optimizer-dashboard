import numpy as np
import pandas as pd

import config as C
import constraints as K
import m1_nutrition as m1
import m2_design as m2
import m3_evaluate as m3
import m4_gpr_df as m4

SAMPLE_SIZES = [100, 300]
OUTPUT_XLSX = "Results.xlsx"


def run():
    K.reset_bounds()
    matrix = m1.load_matrix()
    rows = []
    for n in SAMPLE_SIZES:
        for method in C.DESIGN_METHODS:
            design, info = m2.generate_design(method, n, seed=C.RANDOM_SEED)
            evald = m3.evaluate_design(design, matrix)
            X = design[C.INGREDIENTS].to_numpy(float)
            r2 = m4.loocv_r2(X, evald)
            models = m4.fit_all_responses(X, evald)
            for group in C.PRODUCT_GROUPS:
                best = m4.optimize_desirability(models, group, seed=C.RANDOM_SEED)
                row = {"Group": group, "n requested": n, "Method": method,
                       "Points": info["actual_n"], "Min distance": info["min_distance"],
                       "Mean R2": float(np.nanmean(list(r2.values()))), "D": best["D"]}
                row.update({f"{C.INGREDIENT_NAMES[ing]} (% product)": 100 * C.VARIABLE_SHARE * v
                            for ing, v in zip(C.INGREDIENTS, best["x"])})
                water = m3.responses_from_x(best["x"], matrix)["Water"]
                row.update({f"pred {k}": v for k, v in best["responses"].items()})
                row.update({f"d {k}": v for k, v in best["d"].items()})
                row["Water (g/100 g)"] = water
                rows.append(row)
            print(f"n={n:<4d} {method}")

    results = pd.DataFrame(rows)
    at_300 = results[results["n requested"] == 300]
    best_formula = at_300.loc[at_300.groupby("Group", sort=False)["D"].idxmax()]
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        results.to_excel(w, sheet_name="All runs", index=False)
        best_formula.to_excel(w, sheet_name="Best formula (n=300)", index=False)
        matrix.to_excel(w, sheet_name="Design ingredients", index=False)
        m1.load_raw_matrix().reset_index().to_excel(w, sheet_name="Raw materials (USDA)", index=False)
    return results, best_formula


if __name__ == "__main__":
    run()
    print(f"Saved {OUTPUT_XLSX}")
