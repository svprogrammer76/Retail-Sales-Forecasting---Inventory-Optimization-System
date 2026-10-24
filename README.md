# Retail Sales Forecasting & Inventory Optimization System

A reproducible retail planning simulation that forecasts daily demand for each store-SKU pair and converts predictions into replenishment suggestions. This portfolio project uses synthetic data; it has no access to private retailer systems and does not claim measured company savings.

## Problem and business context

Retailers balance product availability against cash tied up in inventory. Forecasts can inform demand planning, while safety stock and reorder points help planners reason about availability and buffers. This project demonstrates the data and operations workflow in a transparent virtual setting.

## What it does

- Simulates daily demand for 8 products across 3 stores (21,600 rows by default), with weekday and annual patterns, trends, promotions, random variation, and some censored stockout sales.
- Cleans the sales panel and builds calendar, lag and rolling features.
- Fits a Random Forest and evaluates on a chronological 28-day holdout using MAE and RMSE.
- Recursively forecasts the next 28 days, then calculates safety stock, reorder point, EOQ and replenishment alerts.
- Saves tables and charts and provides an interactive Streamlit dashboard.

## Architecture

```text
Synthetic daily retail data → validation → lag/calendar features → chronological evaluation
 → 28-day store-SKU forecast → safety stock + ROP + EOQ → CSV/PNG outputs → Streamlit app
```

## Technology

Python 3.10–3.12, pandas, NumPy, scikit-learn, SciPy, Matplotlib and Streamlit. No GPU is required. Option A for a very easy baseline is a seasonal-naive/rolling forecast with pandas; Option B, implemented here, uses a Random Forest and Streamlit; Option C could add LightGBM/XGBoost, rolling-origin backtests, APIs and a database. Option B is a practical student portfolio choice.

## Quick start

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.pipeline
streamlit run app/streamlit_app.py
```

Mac/Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.pipeline
streamlit run app/streamlit_app.py
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process Bypass` for the current terminal, or invoke `.venv\Scripts\python.exe` directly. Confirm setup with `python -c "import pandas, sklearn, streamlit; print('Setup OK')"`.

## Data and simulation

The generated record grain is date × store × SKU. Fields include category, unit price, promotion, simulated latent demand, observed units sold and a stockout flag. Each product has a demand level; store multipliers, weekly/annual patterns, a small trend, promotion lift and Poisson variation create demand. On some stockout days, sales are censored below latent demand. The model deliberately trains on observed sales, so stockout censoring can bias demand estimates downward. Replace simulated inventory, lead times and costs with verified operational inputs before using the policy for real decisions.

If `data/raw/retail_sales.csv` exists, the pipeline reads it; otherwise it generates the default simulation. Use ISO dates and the columns created by `src/simulate.py` for the simplest compatible input.

## Outputs

- `data/processed/clean_sales.csv`
- `outputs/tables/holdout_predictions.csv`
- `outputs/tables/daily_forecast.csv`
- `outputs/tables/replenishment_recommendations.csv`
- `outputs/figures/sales_trend.png`, `category_sales.png`, `actual_vs_predicted.png`

MAE is average absolute units error per store-SKU-day; RMSE penalizes larger errors more strongly. Compare with a seasonal-naive benchmark before deciding whether the model is useful. The holdout score is not a business outcome. Safety stock uses a normal z-score and recent daily variability; reorder point is lead-time demand plus safety stock. EOQ assumes stable demand and simplified ordering/holding costs. Current on-hand is a labeled demonstration proxy.

## Dashboard

Run `streamlit run app/streamlit_app.py`. The app shows the saved recommendation table, reorder alerts, forecast selection and evaluation chart. Run the pipeline or click the refresh button to create outputs.

## Repository structure

```text
app/          Streamlit dashboard
data/raw/     Optional source CSV (generated data ignored by Git)
data/processed/ Cleaned data output
docs/         Detailed beginner project and proof guide
images/       Curated screenshots for README
notebooks/    Optional exploratory work
outputs/      Generated tables and figures
src/          Simulation, forecasting, inventory and pipeline modules
main.py       Simple pipeline entry point
requirements.txt
```

See [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) for the full walkthrough, publishing steps, proof plan, resume/interview material and troubleshooting.

## Screenshots

Add screenshots from your own run under `images/`; suggested files include `dataset_preview.png`, `sales_trend.png`, `forecast_holdout.png`, `replenishment_table.png` and `dashboard.png`.

Generated example charts:

![Sales trend](outputs/figures/sales_trend.png)
![Holdout forecast](outputs/figures/actual_vs_predicted.png)

```markdown
![Forecast evaluation](images/forecast_holdout.png)
![Dashboard](images/dashboard.png)
```

## Future work and learning outcomes

Benchmark seasonal-naive forecasts; use rolling-origin backtests and probabilistic predictions; address censored demand and intermittent SKUs; model promotion, price elasticity and weather; use actual inventory positions, supplier lead times, minimum order quantities and pack sizes; add a database/API and scenario dashboard. Skills demonstrated include time-series features and validation, simulation, forecast metrics, inventory formulas, modular Python, reporting and dashboarding.

## Author

sHRADDHA VERMA, B.TECH IN COMPUTER SCIENCE
