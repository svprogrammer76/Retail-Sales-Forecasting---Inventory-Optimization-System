"""Leakage-aware global Random Forest with chronological holdout."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = ["sku_code", "store_code", "dayofweek", "month", "dayofyear", "is_weekend", "promotion", "lag_7", "lag_14", "lag_28", "rolling_7", "rolling_28"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.sort_values(["store_id", "sku_id", "date"]).copy()
    x["sku_code"] = x["sku_id"].astype("category").cat.codes
    x["store_code"] = x["store_id"].astype("category").cat.codes
    x["dayofweek"], x["month"], x["dayofyear"] = x.date.dt.dayofweek, x.date.dt.month, x.date.dt.dayofyear
    x["is_weekend"] = (x.date.dt.dayofweek >= 5).astype(int)
    group = x.groupby(["store_id", "sku_id"], observed=True)["units_sold"]
    for lag in (7, 14, 28):
        x[f"lag_{lag}"] = group.shift(lag)
    grp = x.groupby(["store_id", "sku_id"], observed=True).units_sold
    x["rolling_7"] = grp.transform(lambda s: s.shift(1).rolling(7).mean())
    x["rolling_28"] = grp.transform(lambda s: s.shift(1).rolling(28).mean())
    return x.dropna(subset=FEATURES).reset_index(drop=True)


def fit_and_forecast(history: pd.DataFrame, horizon: int = 28):
    feat = make_features(history)
    cutoff = feat.date.max() - pd.Timedelta(days=horizon)
    train, test = feat[feat.date <= cutoff], feat[feat.date > cutoff]
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, max_features=0.85, n_jobs=1, random_state=42)
    model.fit(train[FEATURES], train.units_sold)
    pred = np.maximum(0, model.predict(test[FEATURES]))
    metrics = {"mae": mean_absolute_error(test.units_sold, pred), "rmse": mean_squared_error(test.units_sold, pred) ** 0.5, "holdout_days": int(test.date.nunique())}
    evaluation = test[["date", "store_id", "sku_id", "units_sold"]].copy()
    evaluation["predicted_units"] = pred
    future_rows, hist = [], history.copy().sort_values("date")
    sku_codes = {v: i for i, v in enumerate(sorted(hist.sku_id.unique()))}
    store_codes = {v: i for i, v in enumerate(sorted(hist.store_id.unique()))}
    for _ in range(horizon):
        date = hist.date.max() + pd.Timedelta(days=1)
        batch = []
        for (store, sku), g in hist.groupby(["store_id", "sku_id"], sort=True):
            vals = g.sort_values("date").units_sold.to_numpy()
            batch.append({"date": date, "store_id": store, "sku_id": sku, "category": g.category.iloc[-1], "unit_price": g.unit_price.iloc[-1], "promotion": 0, "sku_code": sku_codes[sku], "store_code": store_codes[store], "dayofweek": date.dayofweek, "month": date.month, "dayofyear": date.dayofyear, "is_weekend": int(date.dayofweek >= 5), "lag_7": vals[-7], "lag_14": vals[-14], "lag_28": vals[-28], "rolling_7": vals[-7:].mean(), "rolling_28": vals[-28:].mean()})
        b = pd.DataFrame(batch)
        b["predicted_units"] = np.maximum(0, model.predict(b[FEATURES]))
        future_rows.extend(b.to_dict("records"))
        add = b[["date", "store_id", "sku_id", "category", "unit_price", "promotion", "predicted_units"]].rename(columns={"predicted_units": "units_sold"})
        hist = pd.concat([hist, add], ignore_index=True)
    return model, metrics, evaluation, pd.DataFrame(future_rows)
