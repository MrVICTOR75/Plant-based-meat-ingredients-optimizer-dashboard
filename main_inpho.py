import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import config as C
import constraints as K
import m1_nutrition as m1
import m2_design as m2
import m3_evaluate as m3
import m4_gpr_df as m4
import m5_contour as m5

st.set_page_config(page_title="Plant-Based Meat Optimizer", layout="wide", page_icon="leaf")


def xlsx(df):
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def get_matrix():
    return m1.load_matrix()


def run_one(method, n, seed, matrix, xi, group, cost_u):
    design, info = m2.generate_design(method, n, seed=seed)
    evald = m3.evaluate_design(design, matrix)
    X = design[C.INGREDIENTS].to_numpy(float)
    models = m4.fit_all_responses(X, evald, xi=xi)
    return dict(design=design, info=info, evald=evald, r2=m4.loocv_r2(X, evald, xi=xi), models=models,
                best=m4.optimize_desirability(models, group, seed=seed, cost_u=cost_u))


st.title("Optimization of Plant-Based Meat Formulations")

with st.sidebar:
    st.header("Parameters")
    n_points = st.number_input("Number of points (n)", min_value=4, max_value=C.MAX_POINTS,
                               value=C.DEFAULT_N_POINTS, step=1)
    seed = st.number_input("Random seed", min_value=0, value=C.RANDOM_SEED, step=1)

    st.subheader("Ingredients (% of product)")
    st.caption("Fixed: " + ", ".join(f"{k} {100 * v:g}%" for k, v in C.FIXED_COMPONENTS.items()))
    bounds_pct = {ing: st.slider(f"{C.INGREDIENT_NAMES[ing]} (%)", 0.0, 100 * C.VARIABLE_SHARE,
                                 tuple(100 * b for b in C.PRODUCT_BOUNDS[ing]), 0.5, key=f"bd_{ing}")
                  for ing in C.INGREDIENTS}

    st.subheader("Desirability")
    group = st.selectbox("Product group", list(C.PRODUCT_GROUPS))
    cost_u = st.slider("Upper cost bound (THB/100 g)", 5.0, 60.0, float(C.COST_U), 1.0)
    xi = st.number_input("GPR correlation xi", min_value=0.1, value=float(C.GP_XI), step=1.0)

    run = st.button("Run", type="primary", width="stretch")

if run:
    if sum(lo for lo, _ in bounds_pct.values()) > 100 * C.VARIABLE_SHARE:
        st.error("Sum of lower bounds is too high.")
        st.stop()
    K.reset_bounds()
    for ing, (lo, hi) in bounds_pct.items():
        K.set_ingredient_bounds(ing, lo / 100 / C.VARIABLE_SHARE, hi / 100 / C.VARIABLE_SHARE)
    matrix = get_matrix()
    results = {}
    prog = st.progress(0.0)
    for i, meth in enumerate(C.DESIGN_METHODS):
        prog.progress(i / len(C.DESIGN_METHODS), text=meth)
        results[meth] = run_one(meth, int(n_points), int(seed), matrix, xi, group, cost_u)
    prog.empty()
    st.session_state.update(results=results, matrix=matrix, group=group, bounds=K.get_bounds(),
                            best_method=max(results, key=lambda m: results[m]["best"]["D"]))

if "results" not in st.session_state:
    st.info("Set the parameters and click Run.")
    st.stop()

S = st.session_state
results, matrix, best_method = S["results"], S["matrix"], S["best_method"]
best_idx = C.DESIGN_METHODS.index(best_method)
comp_df = pd.DataFrame([{"Method": m, "Points": r["info"]["actual_n"],
                         "Min distance": r["info"]["min_distance"],
                         "Mean R2": float(np.nanmean(list(r["r2"].values()))),
                         "D (overall)": r["best"]["D"]} for m, r in results.items()])

st.success(f"Best method: {best_method} (D = {results[best_method]['best']['D']:.4f})")

tabs = st.tabs(["1) Ingredients", "2) Method Comparison", "3) Point Distribution",
                "4) Nutritional Values", "5) GPR + LOOCV", "6) Best Points", "7) Contour"])


def pick_method(key):
    return st.selectbox("Method", C.DESIGN_METHODS, index=best_idx, key=key)


with tabs[0]:
    st.subheader("Design ingredients (per 100 g)")
    st.dataframe(matrix.round(3), width="stretch", hide_index=True)
    raw = m1.load_raw_matrix().reset_index()
    st.subheader("Raw materials, USDA SR Legacy (per 100 g)")
    st.dataframe(raw, width="stretch", hide_index=True)
    st.download_button("Nutrition_Matrix.xlsx", xlsx(raw), "Nutrition_Matrix.xlsx")

with tabs[1]:
    st.dataframe(comp_df.round({"Min distance": 4, "Mean R2": 4, "D (overall)": 4}),
                 width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    for col, y, color in [(c1, "D (overall)", "seagreen"), (c2, "Mean R2", "steelblue")]:
        d = comp_df.sort_values(y)
        fig = px.bar(d, x=y, y="Method", orientation="h", text=d[y].round(4))
        fig.update_traces(marker_color=color)
        col.plotly_chart(fig, width="stretch", key=f"cmp_{y}")
    st.download_button("Method_Comparison.xlsx", xlsx(comp_df), "Method_Comparison.xlsx")

with tabs[2]:
    st.plotly_chart(m5.design_points_fig(results), width="stretch", key="points")
    vm = pick_method("vm_dist")
    st.dataframe(results[vm]["design"].round(4), width="stretch", height=260)
    st.download_button("Design_Points.xlsx", xlsx(results[vm]["design"]), f"Design_Points_{vm}.xlsx")

with tabs[3]:
    vm = pick_method("vm_nutri")
    V = results[vm]
    st.subheader(f"{vm} - per 100 g of product after processing")
    st.dataframe(V["evald"].drop(columns=["Energy_out (kcal)"]).round(3), width="stretch", height=320)
    st.download_button("Project_Data.xlsx", xlsx(V["evald"]), "Project_Data.xlsx")

with tabs[4]:
    vm = pick_method("vm_gpr")
    r2 = results[vm]["r2"]
    for col, (k, v) in zip(st.columns(len(r2)), r2.items()):
        col.metric(f"R2 {k}", f"{v:.4f}")

with tabs[5]:
    vm = pick_method("vm_best")
    fb = m4.format_best(results[vm]["best"])
    st.metric("D", f"{fb['D']:.4f}")
    cA, cB = st.columns(2)
    form_df = pd.DataFrame(fb["formulation_%"].items(), columns=["Ingredient", "Proportion (%)"])
    cA.plotly_chart(px.pie(form_df, names="Ingredient", values="Proportion (%)", hole=0.35),
                    width="stretch", key=f"pie_{vm}")
    cA.dataframe(form_df, width="stretch", hide_index=True)
    cB.dataframe(pd.DataFrame({"d": fb["individual_d"], "Predicted": fb["responses"]}), width="stretch")

with tabs[6]:
    c1, c2 = st.columns(2)
    with c1:
        vm = pick_method("vm_contour")
    lo, hi = (round(100 * C.VARIABLE_SHARE * b[1], 2) for b in S["bounds"])
    oil = c2.slider("Coconut oil (%)", float(lo), float(hi), float(min(max(5.0, lo), hi)), 0.5)
    st.plotly_chart(m5.response_contours_fig(results[vm]["models"], S["bounds"], oil),
                    width="stretch", key="contour")
