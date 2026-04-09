# ⚡ Energy Usage Analysis — Sprint #3 Applied Data Science

## Problem Statement

Energy providers record electricity consumption at granular intervals but
struggle to communicate usage patterns to stakeholders.  This project
analyses 30 days of half-hourly smart-meter data to **identify peak load
periods, daily consumption cycles, and abnormal consumption spikes**.

---

## Folder Structure

```
energy_project/
├── raw_data/
│   └── energy_usage_raw.csv        # 1 440-row synthetic dataset
├── processed_data/
│   └── cleaned_energy_usage.csv    # After cleaning & outlier flagging
├── notebooks/
│   └── energy_analysis.py          # End-to-end analysis script
├── output/
│   ├── daily_consumption.png       # Chart 1 — daily line chart
│   ├── hourly_avg.png              # Chart 2 — hourly bar chart
│   ├── day_of_week_boxplot.png     # Chart 3 — day-of-week boxplot
│   └── spike_detection.png         # Chart 4 — spike scatter plot
├── index.html                      # Self-contained HTML report
├── generate_data.py                # One-time data generator
└── README.md                       # This file
```

---

## Dataset Description

| Column            | Type     | Description                                  |
|-------------------|----------|----------------------------------------------|
| `timestamp`       | datetime | 30-minute interval timestamp                 |
| `consumption_kwh` | float    | Electricity consumption in kWh               |
| `meter_id`        | string   | Meter identifier (MTR-001 / 002 / 003)       |
| `location`        | string   | Building name (Building-A / B / C)           |
| `temperature_c`   | float    | Ambient temperature in °C                    |
| `is_weekend`      | int      | 1 = Saturday/Sunday, 0 = weekday             |

**Injected characteristics:**
- ~2 % spike anomalies (2.5–4× normal consumption)
- ~1 % missing values (`NaN`)
- Morning peak (06:00–09:00), evening peak (17:00–21:00)
- ~10 % weekend uplift over weekday average

---

## How to Run

### Prerequisites
```
pip install pandas numpy matplotlib seaborn
```

### 1. Generate raw data (one-time)
```bash
python generate_data.py
```

### 2. Run the analysis pipeline
```bash
cd notebooks
python energy_analysis.py
```

This single command will:
1. Load & explore the raw CSV (Week 1)
2. Clean data — parse timestamps, interpolate missing values, flag IQR
   outliers, save cleaned CSV (Week 2)
3. Generate 4 charts in `output/` (Week 3)
4. Build `index.html` with embedded base64 chart images

### 3. View the report
Open `index.html` in any modern browser.

---

## Key Findings

| Metric             | Value        |
|--------------------|--------------|
| Total kWh (30 d)   | ~2 200       |
| Avg daily kWh      | ~73          |
| Peak hour          | ~19:00       |
| Spikes detected    | ~30–40       |
| Weekend uplift     | ~+10 %       |

> Exact values depend on the random seed and are computed dynamically during
> the pipeline run.

---

## Tools Used

| Tool        | Version | Purpose                             |
|-------------|---------|-------------------------------------|
| Python      | 3.9+    | Core language                       |
| Pandas      | 1.5+    | Data loading, cleaning, aggregation |
| NumPy       | 1.23+   | Numerical operations & randomness   |
| Matplotlib  | 3.6+    | Chart generation                    |
| Seaborn     | 0.12+   | Statistical box-plot styling        |

---

## License

This project is for educational purposes (Sprint #3 coursework).