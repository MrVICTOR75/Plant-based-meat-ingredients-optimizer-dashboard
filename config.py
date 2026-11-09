import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

USDA_SR_LEGACY_ZIP = os.path.join(BASE_DIR, "FoodData_Central_sr_legacy_food_csv_2018-04.zip")
NUTRITION_MATRIX_XLSX = os.path.join(BASE_DIR, "Nutrition_Matrix.xlsx")

RAW_FOODS = {
    "Soy protein isolate": 174276,
    "Wheat gluten":        168147,
    "Coconut oil":         171412,
    "Agar":                170090,
    "Corn starch":         169698,
}

NUTRIENT_IDS = {1003: "Protein (g)", 1008: "Energy (kcal)", 1004: "Fat (g)",
                1005: "Carb (g)", 1051: "Water (g)"}
NUTRIENTS = list(NUTRIENT_IDS.values())

THB_PER_USD = 33.0

PRICE_USD_PER_KG = {
    "Soy protein isolate": 4.63,
    "Wheat gluten":        2.02,
    "Coconut oil":         1.52,
    "Agar":                8.31,
    "Corn starch":         0.43,
    "Methylcellulose":    10.80,
}


def price_baht_per_100g(item):
    return PRICE_USD_PER_KG[item] * 0.1 * THB_PER_USD


INGREDIENTS = ["Dough", "Oil", "Gel", "Starch"]
D = len(INGREDIENTS)

INGREDIENT_NAMES = {
    "Dough":  "Meat dough",
    "Oil":    "Coconut oil",
    "Gel":    "1% agar gel",
    "Starch": "Corn starch",
}

RECIPES = {
    "Dough":  {"Soy protein isolate": 5.0, "Wheat gluten": 30.0, "Coconut oil": 9.0, "Water": 54.6},
    "Oil":    {"Coconut oil": 100.0},
    "Gel":    {"Agar": 1.0, "Water": 99.0},
    "Starch": {"Corn starch": 100.0},
}

PRODUCT_BOUNDS = {
    "Dough":  (0.05, 0.88),
    "Oil":    (0.05, 0.35),
    "Gel":    (0.00, 0.40),
    "Starch": (0.00, 0.13),
}

FIXED_COMPONENTS = {
    "Methylcellulose":  0.020,
    "Flavoring agent":  0.059,
    "Heme protein":     0.012,
}

VARIABLE_SHARE = 1.0 - sum(FIXED_COMPONENTS.values())

INGREDIENT_BOUNDS = {k: (lo / VARIABLE_SHARE, hi / VARIABLE_SHARE)
                     for k, (lo, hi) in PRODUCT_BOUNDS.items()}

PROCESS_LOSS = {
    "Protein (g)":   0.10,
    "Fat (g)":       0.15,
    "Carb (g)":      0.05,
    "Energy (kcal)": 0.00,
}
KJ_PER_KCAL = 4.184

PRODUCT_GROUPS = {
    "Burger":      {"Energy": (355, 1160), "Protein": (2.9, 20.9), "Fat": (0.5, 17.7), "Carb": (3.9, 33.0)},
    "Ground meat": {"Energy": (312,  950), "Protein": (4.6, 23.4), "Fat": (0.5, 15.0), "Carb": (1.6, 27.0)},
    "Sausage":     {"Energy": (458, 1103), "Protein": (2.8, 23.0), "Fat": (1.0, 19.0), "Carb": (0.5, 24.4)},
    "Chicken":     {"Energy": (274, 1130), "Protein": (1.2, 36.1), "Fat": (2.0, 14.3), "Carb": (2.7, 21.5)},
    "Seafood":     {"Energy": (231, 1178), "Protein": (0.3, 14.0), "Fat": (0.5, 12.6), "Carb": (4.4, 30.0)},
    "Other":       {"Energy": (405, 1180), "Protein": (5.7, 26.4), "Fat": (1.0, 23.2), "Carb": (1.3, 26.7)},
}

COST_U = 30.0

GP_XI     = 10.0
GP_NUGGET = 1e-8

DEFAULT_N_POINTS = 100
RANDOM_SEED      = 42
MAX_POINTS       = 500

MAXENT_CANDIDATE_POOL  = 2000
MORRIS_MITCHELL_LAMBDA = 40.0
MAXIMIN_REFINE_MAX_N   = 400

DESIGN_METHODS = [
    "Latin Hypercube (Rejection)",
    "Latin Hypercube (Transformation)",
    "Simplex Lattice",
    "Simplex Centroid",
    "Maximin (Morris-Mitchell)",
    "Maximum Entropy",
    "Dirichlet",
]
