"""Transparent continuous-review replenishment suggestions."""
import numpy as np
import pandas as pd
from scipy.stats import norm

def recommend(forecast, history, service_level=0.95, lead_time_days=7, review_days=7, order_cost=50.0, annual_holding_rate=0.24):
    z = norm.ppf(service_level)
    rows = []
    for (store, sku), f in forecast.groupby(["store_id", "sku_id"]):
        h = history[(history.store_id == store) & (history.sku_id == sku)].sort_values("date")
        demand = h.units_sold.tail(90).astype(float)
        avg, sd = float(f.predicted_units.mean()), float(demand.std(ddof=1) or 0)
        safety = z * sd * np.sqrt(lead_time_days)
        rop = avg * lead_time_days + safety
        annual = max(avg * 365, 1)
        unit_cost = float(h.unit_price.iloc[-1]) * 0.6
        eoq = np.sqrt(2 * annual * order_cost / max(unit_cost * annual_holding_rate, 0.01))
        target = avg * (lead_time_days + review_days) + safety
        on_hand = float(demand.tail(14).sum() * 0.5)  # Illustrative proxy, replace with inventory position.
        qty = max(0, int(np.ceil(max(target - on_hand, 0))))
        rows.append({"store_id": store, "sku_id": sku, "category": h.category.iloc[-1], "forecast_daily_units": round(avg, 2), "lead_time_days": lead_time_days, "service_level": service_level, "safety_stock_units": int(np.ceil(safety)), "reorder_point_units": int(np.ceil(rop)), "eoq_units": int(np.ceil(eoq)), "simulated_on_hand_units": int(on_hand), "recommended_order_units": qty, "reorder_now": on_hand <= rop})
    return pd.DataFrame(rows).sort_values(["reorder_now", "recommended_order_units"], ascending=[False, False])
