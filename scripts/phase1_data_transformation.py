"""Phase 1 - Data Transformation (Module 02)."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Metro_Interstate_Traffic_Volume.csv"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "phase1"
PROC.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


section("1. LOAD RAW DATA")
# "None" in the holiday column means no holiday, so don't let pandas turn it into NaN
df = pd.read_csv(RAW, keep_default_na=False)
df["date_time"] = pd.to_datetime(df["date_time"])
print(f"Shape : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Period: {df.date_time.min()} -> {df.date_time.max()}")
print("\nFirst rows:\n" + df.head().to_string())
print("\nColumn types:\n" + df.dtypes.to_string())
print("\nSummary of numeric columns:\n" + df.describe().round(2).to_string())


section("2. DATA DEDUPLICATION")
n_before = len(df)
n_exact = df.duplicated().sum()
df = df.drop_duplicates()
print(f"Exact duplicate rows removed: {n_exact}")

n_same_hour = df.date_time.duplicated().sum()
print(f"Rows repeating an hour already in the data: {n_same_hour}")
example = df[df.date_time.duplicated(keep=False)].head(3)
print(example[["date_time", "weather_main", "weather_description", "traffic_volume"]]
      .to_string(index=False))
print("An hour is listed once per weather condition with the same traffic volume,"
      "\nso only the first row of each hour is kept.")
df = df.drop_duplicates(subset="date_time", keep="first").sort_values("date_time")
print(f"Rows: {n_before} -> {len(df)}")

# holiday name only appears on the 00:00 row, copy it to the rest of the day
hol_rows = df.loc[df.holiday != "None"]
hol_by_date = pd.Series(hol_rows.holiday.values, index=hol_rows.date_time.dt.date)
df["holiday"] = df.date_time.dt.date.map(hol_by_date).fillna("None")
print(f"Holiday hours after filling the whole day: {(df.holiday != 'None').sum()} "
      f"({hol_by_date.size} dates)")


section("3. HANDLING MISSING DATA")
print("NaN values per column:")
print(df.isna().sum().to_string())
bad_temp = df.temp == 0
bad_rain = df.rain_1h > 1000
print(f"\nImpossible values treated as missing:")
print(f"  temp = 0 K               : {bad_temp.sum()} rows")
print(f"  rain_1h > 1000 mm in 1 h : {bad_rain.sum()} row (value {df.rain_1h.max():.0f})")
df.loc[bad_temp, "temp"] = np.nan
df.loc[bad_rain, "rain_1h"] = np.nan

deleted = df.dropna()
imputed = df.copy()
imputed["temp"] = imputed.temp.fillna(imputed.temp.median())
imputed["rain_1h"] = imputed.rain_1h.fillna(imputed.rain_1h.median())
cmp = pd.DataFrame({
    "rows kept": [len(deleted), len(imputed)],
    "temp mean (K)": [deleted.temp.mean(), imputed.temp.mean()],
    "temp median (K)": [deleted.temp.median(), imputed.temp.median()],
    "temp std": [deleted.temp.std(), imputed.temp.std()],
    "traffic mean": [deleted.traffic_volume.mean(), imputed.traffic_volume.mean()],
}, index=["Deletion", "Median imputation"])
print("\nDeletion vs median imputation:")
print(cmp.round(3).to_string())
print(f"Deletion drops {len(df) - len(deleted)} hours of traffic data; median imputation keeps"
      "\nthem and leaves the statistics practically unchanged, so it is used.")
print(f"Mode for categorical columns (weather_main): '{df.weather_main.mode()[0]}' "
      "(no categorical values are missing)")
df = imputed

print(f"\nHours with traffic_volume = 0: {(df.traffic_volume == 0).sum()} "
      "(kept, checked again in outlier detection)")
full_hours = pd.date_range(df.date_time.min(), df.date_time.max(), freq="h")
print(f"Hours missing from the timeline: {len(full_hours) - len(df)} of {len(full_hours)} "
      f"({1 - len(df) / len(full_hours):.0%}), mostly the Aug 2014 - Jun 2015 sensor outage."
      "\nThese are not filled in.")


section("4. DISCRETIZATION AND BINNING")
df["temp_c"] = (df.temp - 273.15).round(2)

LOW_MAX, HIGH_MIN = 2000, 4500
df["traffic_level"] = pd.cut(df.traffic_volume, bins=[-1, LOW_MAX - 1, HIGH_MIN - 1, np.inf],
                             labels=["Low", "Medium", "High"])
t1, t2 = df.traffic_volume.quantile([1 / 3, 2 / 3])
print(f"Traffic level: Low < {LOW_MAX}, Medium {LOW_MAX}-{HIGH_MIN - 1}, High >= {HIGH_MIN} veh/h")
print(f"(tertiles of the data are {t1:.0f} and {t2:.0f})")
print("This is a volume-based congestion proxy; the data has no speed or delay measurements.")
print(df.traffic_level.value_counts().reindex(["Low", "Medium", "High"]).to_string())

df["temp_band"] = pd.cut(df.temp_c, bins=[-np.inf, 0, 15, np.inf],
                         labels=["Cold (<0 C)", "Mild (0-15 C)", "Warm (>15 C)"])
print("\n" + df.temp_band.value_counts().sort_index().to_string())

df["hour"] = df.date_time.dt.hour
df["time_of_day"] = pd.cut(df.hour, bins=[-1, 5, 9, 15, 19, 23],
                           labels=["Night (0-5)", "AM peak (6-9)", "Midday (10-15)",
                                   "PM peak (16-19)", "Evening (20-23)"])
print("\n" + df.time_of_day.value_counts().sort_index().to_string())

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.hist(df.traffic_volume, bins=70, color="#9fb7d4")
for x in (LOW_MAX, HIGH_MIN):
    ax.axvline(x, color="#b5332e", ls="--")
for x, label in [(LOW_MAX / 2, "Low"), ((LOW_MAX + HIGH_MIN) / 2, "Medium"), (5900, "High")]:
    ax.text(x, ax.get_ylim()[1] * 0.92, label, ha="center", fontsize=12, color="#b5332e")
ax.set_xlabel("traffic volume (vehicles per hour)")
ax.set_ylabel("number of hours")
ax.set_title("Traffic volume with Low / Medium / High thresholds")
fig.tight_layout()
fig.savefig(OUT / "01_traffic_level_bins.png", dpi=150)
plt.close(fig)


section("5. SAVE CLEANED DATASET")
df["year"] = df.date_time.dt.year
df["month"] = df.date_time.dt.month
df["day_of_week"] = df.date_time.dt.day_of_week
df["day_name"] = df.date_time.dt.day_name()
df["is_weekend"] = df.day_of_week >= 5
df["is_holiday"] = df.holiday != "None"
df["day_type"] = np.select([df.is_holiday, df.is_weekend], ["Holiday", "Weekend"],
                           default="Weekday")
df["season"] = df.month.map({12: "Winter", 1: "Winter", 2: "Winter",
                             3: "Spring", 4: "Spring", 5: "Spring",
                             6: "Summer", 7: "Summer", 8: "Summer",
                             9: "Autumn", 10: "Autumn", 11: "Autumn"})

clean = df[["date_time", "traffic_volume", "traffic_level", "temp_c", "temp_band",
            "rain_1h", "snow_1h", "clouds_all", "weather_main", "holiday", "is_holiday",
            "year", "month", "season", "day_of_week", "day_name", "is_weekend",
            "day_type", "hour", "time_of_day"]].reset_index(drop=True)
clean.to_csv(PROC / "traffic_clean.csv", index=False)
clean.to_pickle(PROC / "traffic_clean.pkl")
print(f"Cleaned dataset: {clean.shape[0]} rows x {clean.shape[1]} columns")
print("Saved to data/processed/traffic_clean.csv and traffic_clean.pkl")
print("\n" + clean.day_type.value_counts().to_string())
print(f"\nRemaining missing values: {int(clean.isna().sum().sum())}")
