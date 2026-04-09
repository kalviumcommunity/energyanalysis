import pandas as pd
import numpy as np
import os

# Ensure directories exist
os.makedirs('../raw_data', exist_ok=True)
os.makedirs('../processed_data', exist_ok=True)
os.makedirs('../output', exist_ok=True)

# Generate 30 days of data at 30 min intervals (1440 rows)
np.random.seed(42)
dates = pd.date_range(start='2024-03-01', periods=1440, freq='30min')
df = pd.DataFrame({'timestamp': dates})

df['meter_id'] = 'MTR-8374'
df['location'] = 'Campus_Bldg_A'
df['is_weekend'] = df['timestamp'].dt.dayofweek >= 5

# Base temp
df['temperature_c'] = np.sin(df['timestamp'].dt.hour / 24 * 2 * np.pi - np.pi/2) * 5 + 15 + np.random.normal(0, 1, 1440)

# Base consumption
hour = df['timestamp'].dt.hour
minute = df['timestamp'].dt.minute
time_decimal = hour + minute / 60.0

consumption = np.zeros(1440)

# Morning peak (6-9 AM), center ~7:30
morning = np.exp(-((time_decimal - 7.5)**2) / 2) * 40
# Evening peak (5-9 PM), center ~19:00
evening = np.exp(-((time_decimal - 19.0)**2) / 4) * 50
# Night base
base = 20 + np.random.normal(0, 2, 1440)

consumption = base + morning + evening

# Weekend uplift ~10%
weekend_mask = df['is_weekend']
consumption[weekend_mask] *= 1.10

# Temperature effect (more correlation with cold or hot)
consumption += (df['temperature_c'] - 15) * 1.5

# Ensure no negative
consumption = np.maximum(consumption, 5)

df['consumption_kwh'] = consumption

# Inject ~2% spike anomalies (2.5-4x normal)
num_anomalies = int(len(df) * 0.02)
anomaly_indices = np.random.choice(df.index, num_anomalies, replace=False)
df.loc[anomaly_indices, 'consumption_kwh'] *= np.random.uniform(2.5, 4.0, num_anomalies)

# Inject duplicates
duplicate_indices = np.random.choice(df.index, 5, replace=False)
duplicates = df.loc[duplicate_indices].copy()
df = pd.concat([df, duplicates]).sort_index()

# Inject ~1% missing values
num_missing = int(len(df) * 0.01)
missing_indices = np.random.choice(df.index, num_missing, replace=False)
df.loc[missing_indices, 'consumption_kwh'] = np.nan

# Save raw dataset
df.to_csv('../raw_data/energy_usage_raw.csv', index=False)
print("Generated raw_data/energy_usage_raw.csv")
