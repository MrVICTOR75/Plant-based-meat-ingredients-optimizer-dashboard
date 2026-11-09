import os
import zipfile
import pandas as pd

import config as C


def _read_member(z, suffix, **kwargs):
    name = next(n for n in z.namelist() if n.endswith(suffix))
    with z.open(name) as f:
        return pd.read_csv(f, low_memory=False, **kwargs)


def build_raw_matrix(save=True):
    ids = set(C.RAW_FOODS.values())
    with zipfile.ZipFile(C.USDA_SR_LEGACY_ZIP) as z:
        foods = _read_member(z, "/food.csv", usecols=["fdc_id", "description"])
        nutr = _read_member(z, "/food_nutrient.csv", usecols=["fdc_id", "nutrient_id", "amount"])
    foods = foods[foods["fdc_id"].isin(ids)].set_index("fdc_id")["description"]
    nutr = nutr[nutr["fdc_id"].isin(ids) & nutr["nutrient_id"].isin(C.NUTRIENT_IDS)]
    nutr = nutr.pivot_table(index="fdc_id", columns="nutrient_id", values="amount", aggfunc="first")

    rows = []
    for item, fdc_id in C.RAW_FOODS.items():
        row = {"Item": item, "Source": "USDA FoodData Central (SR Legacy)",
               "fdc_id": fdc_id, "USDA description": foods[fdc_id]}
        for nid, col in C.NUTRIENT_IDS.items():
            row[col] = float(nutr.loc[fdc_id, nid]) if nid in nutr.columns else 0.0
        row["Price (USD/kg)"] = C.PRICE_USD_PER_KG[item]
        rows.append(row)
    raw = pd.DataFrame(rows)
    if save:
        raw.to_excel(C.NUTRITION_MATRIX_XLSX, index=False, engine="openpyxl")
    return raw


def load_raw_matrix():
    raw = pd.read_excel(C.NUTRITION_MATRIX_XLSX)
    raw["Price (USD/kg)"] = raw["Item"].map(C.PRICE_USD_PER_KG)
    return raw.set_index("Item")


def load_matrix():
    raw = load_raw_matrix()
    rows = []
    for ing in C.INGREDIENTS:
        recipe = C.RECIPES[ing]
        row = {"Category": ing, "Ingredient": C.INGREDIENT_NAMES[ing]}
        for col in C.NUTRIENTS:
            row[col] = sum(g / 100.0 * raw.loc[item, col]
                           for item, g in recipe.items() if item != "Water")
        row["Water (g)"] += recipe.get("Water", 0.0)
        row["Cost (Baht)"] = sum(g / 100.0 * C.price_baht_per_100g(item)
                                 for item, g in recipe.items() if item != "Water")
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_cost_baht():
    return C.FIXED_COMPONENTS["Methylcellulose"] * C.price_baht_per_100g("Methylcellulose")


if __name__ == "__main__":
    if not os.path.exists(C.USDA_SR_LEGACY_ZIP):
        raise SystemExit("Missing " + C.USDA_SR_LEGACY_ZIP)
    print(build_raw_matrix().to_string(index=False))
    print(load_matrix().round(3).to_string(index=False))
