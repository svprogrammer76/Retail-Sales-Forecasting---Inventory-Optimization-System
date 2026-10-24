"""Run the simulation, forecast and replenishment workflow; save tables and charts."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless-safe file rendering for batch runs.
import matplotlib.pyplot as plt
from src.config import RAW, PROCESSED, OUTPUTS
from src.simulate import generate_sales
from src.forecast import fit_and_forecast
from src.inventory import recommend

def run():
    RAW.parent.mkdir(parents=True, exist_ok=True); PROCESSED.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "tables").mkdir(parents=True, exist_ok=True); (OUTPUTS / "figures").mkdir(parents=True, exist_ok=True)
    df = generate_sales() if not RAW.exists() else pd.read_csv(RAW, parse_dates=["date"])
    df["date"] = pd.to_datetime(df.date, errors="coerce")
    df = df.dropna(subset=["date", "store_id", "sku_id", "units_sold"])
    df["units_sold"] = pd.to_numeric(df.units_sold, errors="coerce").clip(lower=0)
    df = df.dropna(subset=["units_sold"]).sort_values(["store_id", "sku_id", "date"])
    df.to_csv(PROCESSED / "clean_sales.csv", index=False)
    _, metrics, evaluation, forecast = fit_and_forecast(df)
    inventory = recommend(forecast, df)
    evaluation.to_csv(OUTPUTS / "tables" / "holdout_predictions.csv", index=False)
    forecast.to_csv(OUTPUTS / "tables" / "daily_forecast.csv", index=False)
    inventory.to_csv(OUTPUTS / "tables" / "replenishment_recommendations.csv", index=False)
    daily = df.groupby("date").units_sold.sum()
    plt.figure(figsize=(11, 4)); plt.plot(daily.index, daily.values); plt.title("Simulated daily retail sales"); plt.ylabel("Units sold"); plt.tight_layout(); plt.savefig(OUTPUTS / "figures" / "sales_trend.png", dpi=150); plt.close()
    actual, pred = evaluation.groupby("date").units_sold.sum(), evaluation.groupby("date").predicted_units.sum()
    plt.figure(figsize=(11, 4)); plt.plot(actual.index, actual.values, label="Actual"); plt.plot(pred.index, pred.values, label="Forecast"); plt.legend(); plt.title("Chronological holdout: actual vs predicted"); plt.tight_layout(); plt.savefig(OUTPUTS / "figures" / "actual_vs_predicted.png", dpi=150); plt.close()
    df.groupby("category").units_sold.sum().sort_values().plot(kind="barh", figsize=(8, 4), title="Units sold by category")
    plt.tight_layout(); plt.savefig(OUTPUTS / "figures" / "category_sales.png", dpi=150); plt.close()
    pairs = df.groupby(["store_id", "sku_id"]).ngroups
    print(f"Rows: {len(df):,} | SKU-store pairs: {pairs} | Holdout MAE: {metrics['mae']:.2f} units | RMSE: {metrics['rmse']:.2f} units")
    print(f"Reorder alerts: {int(inventory.reorder_now.sum())} | Forecast horizon: {forecast.date.nunique()} days")
    print(f"Outputs saved under {OUTPUTS}")
    return metrics, inventory

if __name__ == "__main__": run()
