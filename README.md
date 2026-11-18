# Plant-Based Meat Formulation Optimizer

Interactive dashboard: https://plant-meat-ingredients-optimizer.streamlit.app

## Running

Python 3.13 (versions pinned in `requirements.txt`).

```bash
pip install -r requirements.txt
streamlit run main_inpho.py
```

Reproduce the results into `Results.xlsx` (seed 42 for the main runs, seeds 1-30 for the regret summary, n = 100 and 300, all methods and product groups):

```bash
python reproduce_results.py
```

Figures 1 and 2 are drawn by `m6_figures.py` (300 dpi PNG; Figure 2 uses the Maximin design, n = 300, coconut oil 5%).

Rebuild `Nutrition_Matrix.xlsx` from the raw USDA data: download the SR Legacy CSV archive (2018-04) from https://fdc.nal.usda.gov/download-datasets, place the zip file next to the code, and run

```bash
python m1_nutrition.py
```

## License

MIT
