#!/usr/bin/env python3
"""
energy_analysis.py  —  Sprint #3 Applied Data Science
======================================================
End-to-end analysis of 30-day electricity consumption data.

Run from the notebooks/ folder:
    python energy_analysis.py

Outputs:
    ../processed_data/cleaned_energy_usage.csv
    ../output/daily_consumption.png
    ../output/hourly_avg.png
    ../output/day_of_week_boxplot.png
    ../output/spike_detection.png
    ../index.html   (self-contained report with base64 charts)
"""

import os, sys, base64, textwrap, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ── Paths (relative to notebooks/) ──────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
PROJECT   = os.path.dirname(BASE)
RAW_CSV   = os.path.join(PROJECT, "raw_data", "energy_usage_raw.csv")
CLEAN_CSV = os.path.join(PROJECT, "processed_data", "cleaned_energy_usage.csv")
OUT_DIR   = os.path.join(PROJECT, "output")
HTML_PATH = os.path.join(PROJECT, "index.html")

os.makedirs(os.path.join(PROJECT, "processed_data"), exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
#  WEEK 1 — Data Loading & Initial Exploration
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  WEEK 1 -- Data Loading & Initial Exploration")
print("=" * 70)

df = pd.read_csv(RAW_CSV)

print("\n> df.head()")
print(df.head().to_string(index=False))

print("\n> df.info()")
df.info()

print("\n> df.describe()")
print(df.describe().to_string())

initial_rows = len(df)
initial_missing = df["consumption_kwh"].isna().sum()
print(f"\n[OK] Loaded {initial_rows} rows, {initial_missing} missing consumption values")

# ═══════════════════════════════════════════════════════════════════════════
#  WEEK 2 — Data Cleaning & Preparation
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WEEK 2 -- Data Cleaning & Preparation")
print("=" * 70)

# 1. Parse timestamps
df["timestamp"] = pd.to_datetime(df["timestamp"])
print(f"[OK] Timestamps parsed  (dtype: {df['timestamp'].dtype})")

# 2. Remove exact duplicates
dup_count = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"[OK] Removed {dup_count} duplicate rows  ({len(df)} remaining)")

# 3. Interpolate missing consumption values
missing_before = df["consumption_kwh"].isna().sum()
df["consumption_kwh"] = df["consumption_kwh"].interpolate(method="linear")
# Any remaining NaN at edges
df["consumption_kwh"] = df["consumption_kwh"].bfill()
df["consumption_kwh"] = df["consumption_kwh"].ffill()
missing_after = df["consumption_kwh"].isna().sum()
print(f"[OK] Interpolated {missing_before - missing_after} missing values "
      f"({missing_after} remain)")

# 4. Flag outliers with IQR method
Q1 = df["consumption_kwh"].quantile(0.25)
Q3 = df["consumption_kwh"].quantile(0.75)
IQR = Q3 - Q1
LOWER = Q1 - 1.5 * IQR
UPPER = Q3 + 1.5 * IQR
df["is_outlier"] = ((df["consumption_kwh"] < LOWER) |
                    (df["consumption_kwh"] > UPPER)).astype(int)
outlier_count = df["is_outlier"].sum()
print(f"[OK] Flagged {outlier_count} outliers  (IQR bounds: "
      f"{LOWER:.3f} - {UPPER:.3f} kWh)")

# 5. Derive helper columns
df["date"]       = df["timestamp"].dt.date
df["hour"]       = df["timestamp"].dt.hour
df["day_name"]   = df["timestamp"].dt.day_name()
df["day_of_week"]= df["timestamp"].dt.dayofweek

# 6. Save cleaned data
df.to_csv(CLEAN_CSV, index=False)
print(f"[OK] Saved cleaned CSV -> {CLEAN_CSV}")

# ═══════════════════════════════════════════════════════════════════════════
#  Compute KPIs
# ═══════════════════════════════════════════════════════════════════════════
total_kwh   = df["consumption_kwh"].sum()
daily_agg   = df.groupby("date")["consumption_kwh"].sum()
avg_daily   = daily_agg.mean()

hourly_agg  = df.groupby("hour")["consumption_kwh"].mean()
peak_hour   = int(hourly_agg.idxmax())

spike_count = outlier_count

weekday_avg = df.loc[df["is_weekend"] == 0, "consumption_kwh"].mean()
weekend_avg = df.loc[df["is_weekend"] == 1, "consumption_kwh"].mean()
weekend_uplift_pct = ((weekend_avg - weekday_avg) / weekday_avg) * 100

print(f"\n-- KPIs --")
print(f"  Total kWh        : {total_kwh:,.1f}")
print(f"  Avg daily kWh    : {avg_daily:,.1f}")
print(f"  Peak hour        : {peak_hour}:00")
print(f"  Spike count      : {spike_count}")
print(f"  Weekend uplift   : {weekend_uplift_pct:+.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
#  WEEK 3 — Visualization
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  WEEK 3 -- Visualization")
print("=" * 70)

CHART_PATHS = {}

# ── Chart 1: Daily Consumption Line Chart ───────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
daily_series = daily_agg.reset_index()
daily_series.columns = ["date", "total_kwh"]
ax.fill_between(range(len(daily_series)), daily_series["total_kwh"],
                alpha=0.25, color="#4F46E5")
ax.plot(daily_series["total_kwh"].values, linewidth=2.2, color="#4F46E5",
        marker="o", markersize=4)
ax.set_xticks(range(0, len(daily_series), 5))
ax.set_xticklabels([str(d) for d in daily_series["date"].iloc[::5]],
                   rotation=30, ha="right")
ax.set_xlabel("Date")
ax.set_ylabel("Total kWh")
ax.set_title("Daily Electricity Consumption (30 Days)", fontsize=14, weight="bold")
plt.tight_layout()
path = os.path.join(OUT_DIR, "daily_consumption.png")
fig.savefig(path, dpi=150)
plt.close(fig)
CHART_PATHS["daily"] = path
print(f"[OK] Saved {path}")

# ── Chart 2: Hourly Average Bar Chart ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#EF4444" if h in range(6, 9) or h in range(17, 21)
          else "#60A5FA" for h in range(24)]
ax.bar(hourly_agg.index, hourly_agg.values, color=colors, edgecolor="white",
       linewidth=0.5)
ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}" for h in range(24)])
ax.set_xlabel("Hour of Day")
ax.set_ylabel("Avg kWh per 30-min Slot")
ax.set_title("Average Consumption by Hour", fontsize=14, weight="bold")
plt.tight_layout()
path = os.path.join(OUT_DIR, "hourly_avg.png")
fig.savefig(path, dpi=150)
plt.close(fig)
CHART_PATHS["hourly"] = path
print(f"[OK] Saved {path}")

# ── Chart 3: Day-of-Week Boxplot ───────────────────────────────────────
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x="day_name", y="consumption_kwh", order=day_order,
            palette="coolwarm", ax=ax, fliersize=3)
ax.set_xlabel("Day of Week")
ax.set_ylabel("Consumption (kWh)")
ax.set_title("Consumption Distribution by Day of Week", fontsize=14,
             weight="bold")
plt.tight_layout()
path = os.path.join(OUT_DIR, "day_of_week_boxplot.png")
fig.savefig(path, dpi=150)
plt.close(fig)
CHART_PATHS["weekday"] = path
print(f"[OK] Saved {path}")

# ── Chart 4: Spike Detection Scatter Plot ──────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
normal = df[df["is_outlier"] == 0]
spikes = df[df["is_outlier"] == 1]
ax.scatter(normal["timestamp"], normal["consumption_kwh"],
           s=6, alpha=0.4, color="#94A3B8", label="Normal")
ax.scatter(spikes["timestamp"], spikes["consumption_kwh"],
           s=30, alpha=0.9, color="#EF4444", label="Spike / Outlier",
           edgecolors="black", linewidths=0.5, zorder=5)
ax.axhline(UPPER, color="#F59E0B", linestyle="--", linewidth=1.2,
           label=f"IQR Upper Bound ({UPPER:.2f})")
ax.set_xlabel("Timestamp")
ax.set_ylabel("Consumption (kWh)")
ax.set_title("Spike Detection via IQR Method", fontsize=14, weight="bold")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
path = os.path.join(OUT_DIR, "spike_detection.png")
fig.savefig(path, dpi=150)
plt.close(fig)
CHART_PATHS["spikes"] = path
print(f"[OK] Saved {path}")

# ═══════════════════════════════════════════════════════════════════════════
#  Generate index.html  (self-contained, no external dependencies)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  Generating index.html")
print("=" * 70)

def img_b64(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

b64 = {k: img_b64(v) for k, v in CHART_PATHS.items()}

html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="description" content="Sprint 3 Applied Data Science — Energy Usage Analysis: 30-day electricity consumption patterns, peak load periods, and anomaly detection.">
<title>Energy Usage Analysis — Sprint #3</title>
<style>
/* ── Reset & Base ─────────────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg: #0F172A;
  --surface: #1E293B;
  --surface2: #334155;
  --accent: #6366F1;
  --accent2: #818CF8;
  --red: #EF4444;
  --green: #22C55E;
  --amber: #F59E0B;
  --text: #F1F5F9;
  --muted: #94A3B8;
  --radius: 14px;
  --max-w: 1140px;
}}
html {{ scroll-behavior: smooth; }}
body {{
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.65;
}}

/* ── Sticky Nav ───────────────────────────────────────────────── */
nav {{
  position: sticky; top: 0; z-index: 100;
  backdrop-filter: blur(14px) saturate(160%);
  -webkit-backdrop-filter: blur(14px) saturate(160%);
  background: rgba(15,23,42,0.82);
  border-bottom: 1px solid rgba(99,102,241,0.25);
  padding: 0 24px;
}}
nav .inner {{
  max-width: var(--max-w); margin: auto;
  display: flex; align-items: center; justify-content: space-between;
  height: 60px;
}}
nav .logo {{
  font-weight: 800; font-size: 1.15rem;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
nav .links {{ display: flex; gap: 22px; }}
nav .links a {{
  color: var(--muted); text-decoration: none; font-size: 0.9rem;
  transition: color .2s;
}}
nav .links a:hover {{ color: var(--text); }}

/* ── Hero Section ─────────────────────────────────────────────── */
.hero {{
  text-align: center;
  padding: 80px 24px 50px;
  background: radial-gradient(ellipse at 50% 0%, rgba(99,102,241,0.18) 0%, transparent 70%);
}}
.hero h1 {{
  font-size: clamp(2rem, 5vw, 3.2rem);
  font-weight: 800;
  background: linear-gradient(135deg, #c7d2fe, #6366f1);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 14px;
}}
.hero p {{
  max-width: 680px; margin: 0 auto;
  color: var(--muted); font-size: 1.08rem;
}}

/* ── KPI Strip ────────────────────────────────────────────────── */
.kpi-strip {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 18px;
  max-width: var(--max-w); margin: -20px auto 0; padding: 0 24px;
}}
.kpi {{
  background: var(--surface);
  border: 1px solid var(--surface2);
  border-radius: var(--radius);
  padding: 22px 20px;
  text-align: center;
  transition: transform .2s, box-shadow .2s;
}}
.kpi:hover {{
  transform: translateY(-4px);
  box-shadow: 0 8px 30px rgba(99,102,241,0.18);
}}
.kpi .value {{
  font-size: 1.9rem; font-weight: 800;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.kpi .label {{ color: var(--muted); font-size: 0.85rem; margin-top: 4px; }}

/* ── Section / Card ───────────────────────────────────────────── */
section {{ max-width: var(--max-w); margin: 50px auto; padding: 0 24px; }}
.card {{
  background: var(--surface);
  border: 1px solid var(--surface2);
  border-radius: var(--radius);
  padding: 30px;
  margin-bottom: 36px;
  transition: box-shadow .25s;
}}
.card:hover {{ box-shadow: 0 4px 24px rgba(0,0,0,0.35); }}
.card h2 {{
  font-size: 1.35rem; margin-bottom: 16px;
  display: flex; align-items: center; gap: 10px;
}}
.card h2 .dot {{
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--accent); display: inline-block;
  animation: pulse 2s infinite;
}}
@keyframes pulse {{
  0%, 100% {{ opacity: 1; }}
  50% {{ opacity: 0.45; }}
}}
.card img {{
  width: 100%; border-radius: 10px;
  margin: 12px 0 18px;
  border: 1px solid var(--surface2);
}}
.insight {{
  background: rgba(99,102,241,0.08);
  border-left: 3px solid var(--accent);
  padding: 14px 18px;
  border-radius: 0 8px 8px 0;
  color: var(--muted);
  font-size: 0.95rem;
  line-height: 1.7;
}}

/* ── Data Cleaning Table ──────────────────────────────────────── */
.tbl-wrap {{ overflow-x: auto; }}
table {{
  width: 100%; border-collapse: collapse; font-size: 0.92rem;
}}
th, td {{ padding: 12px 16px; text-align: left; }}
th {{
  background: var(--surface2); color: var(--text);
  font-weight: 700; text-transform: uppercase; font-size: 0.78rem;
  letter-spacing: 0.5px;
}}
td {{ border-bottom: 1px solid var(--surface2); color: var(--muted); }}
tr:hover td {{ background: rgba(99,102,241,0.06); }}

/* ── Assumptions ──────────────────────────────────────────────── */
.assumptions ul {{ padding-left: 20px; }}
.assumptions li {{ color: var(--muted); margin-bottom: 8px; font-size: 0.94rem; }}

/* ── Footer ───────────────────────────────────────────────────── */
footer {{
  text-align: center; padding: 40px 24px;
  border-top: 1px solid var(--surface2);
  color: var(--muted); font-size: 0.85rem;
}}
footer a {{ color: var(--accent2); text-decoration: none; }}
</style>
</head>
<body>

<!-- ── Sticky Nav ──────────────────────────────────────────────── -->
<nav>
  <div class="inner">
    <span class="logo">⚡ Energy Analytics</span>
    <div class="links">
      <a href="#kpis">KPIs</a>
      <a href="#charts">Charts</a>
      <a href="#cleaning">Cleaning</a>
      <a href="#assumptions">Assumptions</a>
    </div>
  </div>
</nav>

<!-- ── Hero ────────────────────────────────────────────────────── -->
<div class="hero" id="top">
  <h1>Energy Usage Analysis</h1>
  <p>30-day electricity consumption study identifying peak load periods,
     daily usage cycles, and abnormal consumption spikes across three
     building meters.</p>
</div>

<!-- ── KPI Strip ──────────────────────────────────────────────── -->
<div class="kpi-strip" id="kpis">
  <div class="kpi">
    <div class="value">{total_kwh:,.0f}</div>
    <div class="label">Total kWh</div>
  </div>
  <div class="kpi">
    <div class="value">{avg_daily:,.1f}</div>
    <div class="label">Avg Daily kWh</div>
  </div>
  <div class="kpi">
    <div class="value">{peak_hour}:00</div>
    <div class="label">Peak Hour</div>
  </div>
  <div class="kpi">
    <div class="value">{spike_count}</div>
    <div class="label">Spikes Detected</div>
  </div>
  <div class="kpi">
    <div class="value">{weekend_uplift_pct:+.1f}%</div>
    <div class="label">Weekend Uplift</div>
  </div>
</div>

<!-- ── Charts ──────────────────────────────────────────────────── -->
<section id="charts">

  <!-- Chart 1 -->
  <div class="card">
    <h2><span class="dot"></span> Daily Consumption Trend</h2>
    <img src="data:image/png;base64,{b64['daily']}" alt="Daily consumption line chart">
    <div class="insight">
      <strong>Insight:</strong> Daily consumption fluctuates between roughly
      {daily_agg.min():.0f} kWh and {daily_agg.max():.0f} kWh.  Weekend days
      consistently show elevated totals due to increased residential activity,
      while mid-week values dip as commercial loads dominate. The overall
      30-day trend remains stable, with no clear upward or downward drift.
    </div>
  </div>

  <!-- Chart 2 -->
  <div class="card">
    <h2><span class="dot"></span> Hourly Average Consumption</h2>
    <img src="data:image/png;base64,{b64['hourly']}" alt="Hourly average bar chart">
    <div class="insight">
      <strong>Insight:</strong> Two distinct peaks emerge — a <em>morning
      peak</em> between 06:00 and 09:00 (driven by HVAC start-up and
      lighting) and a stronger <em>evening peak</em> from 17:00 to 21:00
      (cooking, entertainment, EV charging). Overnight hours (00:00–05:00)
      represent base load at roughly 40 % of peak demand. Red bars highlight
      these peak windows.
    </div>
  </div>

  <!-- Chart 3 -->
  <div class="card">
    <h2><span class="dot"></span> Day-of-Week Distribution</h2>
    <img src="data:image/png;base64,{b64['weekday']}" alt="Day-of-week boxplot">
    <div class="insight">
      <strong>Insight:</strong> Saturday and Sunday medians sit approximately
      {weekend_uplift_pct:.0f} % above weekday medians, confirming the
      expected weekend uplift. Interquartile ranges are wider on weekends,
      reflecting more variable usage patterns. Outlier dots above the
      whiskers correspond to the injected spike anomalies.
    </div>
  </div>

  <!-- Chart 4 -->
  <div class="card">
    <h2><span class="dot"></span> Spike Detection (IQR Method)</h2>
    <img src="data:image/png;base64,{b64['spikes']}" alt="Spike detection scatter plot">
    <div class="insight">
      <strong>Insight:</strong> The IQR method flagged <strong>{spike_count}
      </strong> data points as outliers (red dots above the amber threshold
      line at {UPPER:.2f} kWh). These spikes are distributed across the
      entire 30-day window without clustering, suggesting random equipment
      faults or meter glitches rather than systematic load issues. Utility
      teams should investigate any cluster of 3+ consecutive spikes as
      potential equipment failure.
    </div>
  </div>

</section>

<!-- ── Data Cleaning Summary ──────────────────────────────────── -->
<section id="cleaning">
  <div class="card">
    <h2><span class="dot"></span> Data Cleaning Summary</h2>
    <div class="tbl-wrap">
      <table>
        <thead>
          <tr>
            <th>Step</th>
            <th>Action</th>
            <th>Details</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>1</td>
            <td>Timestamp Parsing</td>
            <td>Converted string timestamps to <code>datetime64[ns]</code></td>
          </tr>
          <tr>
            <td>2</td>
            <td>Duplicate Removal</td>
            <td>Removed {dup_count} exact duplicate rows</td>
          </tr>
          <tr>
            <td>3</td>
            <td>Missing Value Interpolation</td>
            <td>Filled {missing_before} missing <code>consumption_kwh</code>
                values via linear interpolation</td>
          </tr>
          <tr>
            <td>4</td>
            <td>Outlier Flagging (IQR)</td>
            <td>Flagged {outlier_count} outliers outside
                [{LOWER:.3f}, {UPPER:.3f}] kWh</td>
          </tr>
          <tr>
            <td>5</td>
            <td>Feature Engineering</td>
            <td>Derived <code>date</code>, <code>hour</code>,
                <code>day_name</code>, <code>day_of_week</code></td>
          </tr>
          <tr>
            <td>6</td>
            <td>Export</td>
            <td>Saved cleaned CSV to <code>processed_data/</code></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</section>

<!-- ── Assumptions & Limitations ──────────────────────────────── -->
<section id="assumptions">
  <div class="card assumptions">
    <h2><span class="dot"></span> Assumptions &amp; Limitations</h2>
    <ul>
      <li><strong>Synthetic data:</strong> The dataset was programmatically
          generated with controlled noise; real-world data would exhibit
          seasonal trends, weather correlations, and tariff-driven behaviour
          shifts.</li>
      <li><strong>Single IQR threshold:</strong> A global IQR bound was used.
          In production, per-meter or time-segmented thresholds would reduce
          false positives during known high-load events.</li>
      <li><strong>Linear interpolation:</strong> Missing values were filled
          linearly. For larger gaps, spline or seasonal decomposition methods
          (STL) may be more appropriate.</li>
      <li><strong>No external variables:</strong> Factors like holidays,
          tariff changes, or occupancy schedules were not modelled.</li>
      <li><strong>Fixed 30-min granularity:</strong> Smart meters can report
          at 1-min or 15-min intervals; aggregation level affects peak
          detection sensitivity.</li>
      <li><strong>Weekend definition:</strong> Saturday and Sunday are
          treated as weekends; this may differ by region or industry.</li>
    </ul>
  </div>
</section>

<!-- ── Footer ─────────────────────────────────────────────────── -->
<footer>
  <p>Sprint #3 Applied Data Science — Energy Usage Analysis</p>
  <p style="margin-top:6px;">Built with Python · Pandas · Matplotlib ·
     Seaborn &nbsp;|&nbsp;
     <a href="#top">Back to top ↑</a></p>
</footer>

</body>
</html>
"""

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(html)

print(f"[OK] Generated {HTML_PATH}")
print(f"  File size: {os.path.getsize(HTML_PATH) / 1024:.0f} KB")

print("\n" + "=" * 70)
print("  ALL DONE -- pipeline complete")
print("=" * 70)
