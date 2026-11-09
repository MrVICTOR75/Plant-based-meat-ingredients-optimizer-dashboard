# Plant-Based Meat Formulation Optimizer

Interactive dashboard: https://plant-meat-ingredients-optimizer.streamlit.app

## Running

Python 3.10 or later.

```bash
pip install -r requirements.txt
streamlit run main_inpho.py
```

Reproduce the results (seed 42, n = 100 and 300, all methods and product groups) into `Results.xlsx`:

```bash
python reproduce_results.py
```

Rebuild `Nutrition_Matrix.xlsx` from the raw USDA data: download the SR Legacy CSV archive (2018-04) from https://fdc.nal.usda.gov/download-datasets, place the zip file next to the code, and run

```bash
python m1_nutrition.py
```
