from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "retail_sales.csv"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
