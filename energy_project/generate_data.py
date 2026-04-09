"""
generate_data.py
Generates a realistic 30-day electricity consumption dataset with 1,440 rows
(30-minute intervals).  Run once, then use notebooks/energy_analysis.py for
analysis.
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ── Parameters ──────────────────────────────────────────────────────────────
DAYS = 30
INTERVAL_MIN = 30
ROWS = DAYS * (24 * 60 // INTERVAL_MIN)          # 1440
START = pd.Timestamp("2025-03-01 00:00:00")
METERS = ["MTR-001", "MTR-002", "MTR-003"]
LOCATIONS = ["Building-A", "Building-B", "Building-C"]
SPIKE_FRAC = 0.02      # 2 % anomaly spikes
MISSING_FRAC = 0.01     # 1 % missing values
WEEKEND_UPLIFT = 1.10   # 10 % weekend uplift

# ── Generate timestamps ────────────────────────────────────────────────────
timestamps = pd.date_range(start=START, periods=ROWS, freq="30min")

# ── Base consumption profile (kWh per 30-min slot) ─────────────────────────
def base_kwh(hour: float) -> float:
    """Return a realistic base consumption given hour of day."""
    if 0 <= hour < 5:
        return np.random.normal(0.8, 0.15)       # night low
    elif 5 <= hour < 6:
        return np.random.normal(1.1, 0.15)        # early rise
    elif 6 <= hour < 9:
        return np.random.normal(2.2, 0.30)        # ★ morning peak
    elif 9 <= hour < 12:
        return np.random.normal(1.6, 0.20)        # mid-morning
    elif 12 <= hour < 14:
        return np.random.normal(1.8, 0.20)        # lunch
    elif 14 <= hour < 17:
        return np.random.normal(1.5, 0.20)        # afternoon
    elif 17 <= hour < 21:
        return np.random.normal(2.4, 0.35)        # ★ evening peak
    elif 21 <= hour < 23:
        return np.random.normal(1.3, 0.20)        # wind-down
    else:
        return np.random.normal(0.9, 0.15)        # late night


# ── Build rows ──────────────────────────────────────────────────────────────
records = []
for ts in timestamps:
    hour = ts.hour + ts.minute / 60.0
    kwh = max(base_kwh(hour), 0.1)                # floor at 0.1

    # Weekend uplift
    is_wknd = int(ts.dayofweek >= 5)
    if is_wknd:
        kwh *= WEEKEND_UPLIFT

    # Assign meter / location
    idx = np.random.randint(0, len(METERS))
    meter = METERS[idx]
    location = LOCATIONS[idx]

    # Temperature: cooler at night, warmer midday, slight daily noise
    temp = 15 + 8 * np.sin(np.pi * (hour - 6) / 12) + np.random.normal(0, 1.5)
    temp = round(temp, 1)

    records.append([ts, round(kwh, 3), meter, location, temp, is_wknd])

df = pd.DataFrame(records,
                  columns=["timestamp", "consumption_kwh", "meter_id",
                           "location", "temperature_c", "is_weekend"])

# ── Inject 2 % spike anomalies (2.5–4× normal) ────────────────────────────
spike_indices = np.random.choice(df.index, size=int(ROWS * SPIKE_FRAC),
                                 replace=False)
for i in spike_indices:
    multiplier = np.random.uniform(2.5, 4.0)
    df.loc[i, "consumption_kwh"] = round(
        df.loc[i, "consumption_kwh"] * multiplier, 3
    )

# ── Inject 1 % missing values in consumption_kwh ───────────────────────────
missing_indices = np.random.choice(
    df.index.difference(spike_indices),  # avoid overlap with spikes
    size=int(ROWS * MISSING_FRAC),
    replace=False,
)
df.loc[missing_indices, "consumption_kwh"] = np.nan

# ── Save ────────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), "raw_data", "energy_usage_raw.csv")
df.to_csv(out_path, index=False)

print(f"[OK] Generated {len(df)} rows  ->  {out_path}")
print(f"  Spikes injected : {len(spike_indices)}")
print(f"  Missing values  : {df['consumption_kwh'].isna().sum()}")
print(f"  Weekend rows    : {df['is_weekend'].sum()}")
print(f"  Date range      : {df['timestamp'].min()} -> {df['timestamp'].max()}")
