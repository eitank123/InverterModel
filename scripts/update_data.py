"""Refresh the Shiller S&P 500 monthly file from github.com/datasets/s-and-p-500.

    python scripts/update_data.py

The source extends Shiller's data every month with the latest S&P 500 level (FRED), so
re-downloading is what moves the paper accounts forward.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import algotrading as at  # noqa: E402

new = pd.read_csv(at.data.SHILLER_URL)
old = pd.read_csv(at.data.SHILLER_PATH)
if len(new) < len(old) or "SP500" not in new.columns:
    sys.exit(f"refusing to replace {len(old)} rows with a suspicious download of {len(new)} rows")
new.to_csv(at.data.SHILLER_PATH, index=False)
print(f"rows {len(old)} -> {len(new)}; last month {new['Date'].iloc[-1]}")
