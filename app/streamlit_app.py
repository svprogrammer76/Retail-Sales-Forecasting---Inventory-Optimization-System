import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
from src.config import OUTPUTS
from src.pipeline import run

st.set_page_config(page_title="Retail Forecast & Replenishment", layout="wide")
st.title("Retail Sales Forecasting & Inventory Optimization")
st.caption("Virtual portfolio simulation • store-SKU predictions • transparent replenishment policy")
if st.button("Run / refresh simulation"):
    with st.spinner("Generating data, fitting forecast and calculating inventory..."): run()
rpath, fpath = OUTPUTS / "tables" / "replenishment_recommendations.csv", OUTPUTS / "tables" / "daily_forecast.csv"
if not rpath.exists():
    st.info("Run `python -m src.pipeline` or press the button to create outputs.")
    st.stop()
r, f = pd.read_csv(rpath), pd.read_csv(fpath)
a, b, c = st.columns(3)
a.metric("Store-SKU pairs", len(r)); b.metric("Reorder alerts", int(r.reorder_now.sum())); c.metric("Forecast horizon", f.date.nunique(), "days")
st.subheader("Replenishment recommendations")
alerts = st.multiselect("Filter", ["All", "Alerts only"], default=["All"])
st.dataframe(r[r.reorder_now] if "Alerts only" in alerts else r, use_container_width=True, hide_index=True)
st.subheader("Forecast")
sku = st.selectbox("SKU", sorted(f.sku_id.unique())); store = st.selectbox("Store", sorted(f.store_id.unique()))
st.line_chart(f[(f.sku_id == sku) & (f.store_id == store)].set_index("date")[["predicted_units"]])
image = OUTPUTS / "figures" / "actual_vs_predicted.png"
if image.exists(): st.image(str(image), caption="Chronological holdout evaluation")
