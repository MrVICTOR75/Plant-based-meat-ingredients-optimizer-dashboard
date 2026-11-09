import numpy as np

import config as C
import m1_nutrition as m1

_COLS = ["Protein (g)", "Energy (kcal)", "Fat (g)", "Carb (g)", "Water (g)", "Cost (Baht)"]


def _matrix(matrix_df):
    return matrix_df.set_index("Category").loc[C.INGREDIENTS, _COLS].to_numpy(float)


def product_composition(X):
    X = np.atleast_2d(np.asarray(X, float))
    return X * C.VARIABLE_SHARE


def _responses(X, matrix_df):
    P = product_composition(X) @ _matrix(matrix_df)
    prot = P[:, 0] * (1 - C.PROCESS_LOSS["Protein (g)"])
    kcal = P[:, 1] * (1 - C.PROCESS_LOSS["Energy (kcal)"])
    fat = P[:, 2] * (1 - C.PROCESS_LOSS["Fat (g)"])
    carb = P[:, 3] * (1 - C.PROCESS_LOSS["Carb (g)"])
    water = P[:, 4]
    cost = P[:, 5] + m1.fixed_cost_baht()
    return prot, kcal, fat, carb, water, cost


def evaluate_design(design_df, matrix_df):
    X = design_df[C.INGREDIENTS].to_numpy(float)
    prot, kcal, fat, carb, water, cost = _responses(X, matrix_df)
    out = design_df.copy()
    for ing, col in zip(C.INGREDIENTS, product_composition(X).T):
        out[f"{ing} (% product)"] = 100 * col
    out["Protein_out (g)"] = prot
    out["Energy_out (kcal)"] = kcal
    out["Energy_out (kJ)"] = kcal * C.KJ_PER_KCAL
    out["Fat_out (g)"] = fat
    out["Carb_out (g)"] = carb
    out["Water (g)"] = water
    out["Cost (Baht)"] = cost
    return out


def responses_from_x(x, matrix_df):
    prot, kcal, fat, carb, water, cost = (float(v[0]) for v in _responses(x, matrix_df))
    return {"Energy": kcal * C.KJ_PER_KCAL, "Protein": prot, "Fat": fat,
            "Carb": carb, "Water": water, "Cost": cost}

