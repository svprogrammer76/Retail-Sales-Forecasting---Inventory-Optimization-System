"""Generate reproducible daily store-SKU demand for a virtual retailer."""
import numpy as np
import pandas as pd


def generate_sales(days: int = 900, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end=end, periods=days, freq="D")
    products = [("SKU-001", "Beverages", 18.0, 35), ("SKU-002", "Snacks", 12.0, 28),
                ("SKU-003", "Household", 55.0, 12), ("SKU-004", "Personal Care", 42.0, 15),
                ("SKU-005", "Dairy", 8.0, 40), ("SKU-006", "Bakery", 6.0, 32),
                ("SKU-007", "Packaged Food", 25.0, 20), ("SKU-008", "Pet Care", 65.0, 8)]
    stores = [("STORE-01", 1.0), ("STORE-02", 0.78), ("STORE-03", 1.22)]
    rows = []
    for sku, category, price, base in products:
        for store, scale in stores:
            level, trend = base * scale, rng.uniform(-0.00015, 0.0004)
            for i, date in enumerate(dates):
                weekly = 1 + 0.18 * (date.dayofweek in (4, 5))
                annual = 1 + 0.15 * np.sin(2 * np.pi * i / 365.25 - 0.8)
                promo = int(rng.random() < 0.035)
                demand = rng.poisson(max(0, level * (1 + trend * i) * weekly * annual * (1.45 if promo else 1)))
                stockout = int(rng.random() < 0.025)
                sales = int(rng.binomial(demand, 0.55)) if stockout else int(demand)
                rows.append((date, store, sku, category, price, promo, demand, sales, stockout))
    return pd.DataFrame(rows, columns=["date", "store_id", "sku_id", "category", "unit_price", "promotion", "simulated_demand", "units_sold", "stockout_flag"])


if __name__ == "__main__":
    from src.config import RAW
    RAW.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_sales()
    frame.to_csv(RAW, index=False)
    print(f"Saved {len(frame):,} rows to {RAW}")
