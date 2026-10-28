"""Refresh the Shiller S&P 500 monthly file from github.com/datasets/s-and-p-500.

    python scripts/update_data.py

The source extends Shiller's data every month with the latest S&P 500 level (FRED), so
re-downloading is what moves the paper accounts forward. It also saves data/fred_monthly.csv
(10-year yield, CPI, T-bills, corporate profits), which fills the series Shiller stopped
publishing in 2023 (see algotrading.data._extend_with_fred).
"""
import io
import os
import sys
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402

with urllib.request.urlopen(at.data.SHILLER_URL, timeout=30) as resp:
    raw = resp.read()
new = pd.read_csv(io.BytesIO(raw))
old = pd.read_csv(at.data.SHILLER_PATH)
if len(new) < len(old) or "SP500" not in new.columns:
    sys.exit(f"refusing to replace {len(old)} rows with a suspicious download of {len(new)} rows")
with open(at.data.SHILLER_PATH, "wb") as f:      # keep the file byte-for-byte as published
    f.write(raw)
print(f"rows {len(old)} -> {len(new)}; last month {new['Date'].iloc[-1]}")
# FRED series that keep rates, CPI, earnings and CAPE current after Shiller's 2023 cut-off.
fred = at.data.save_fred_monthly()
print("FRED monthly: " + ", ".join(f"{c} to {fred[c].last_valid_index():%Y-%m}" for c in fred))
