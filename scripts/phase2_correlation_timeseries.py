"""Phase 2 - Correlation Analysis and Time Series (Module 03)."""
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase2"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
DAY_COLORS = {"Weekday": "#3b6ea5", "Weekend": "#c96f3b", "Holiday": "#4c9a5f"}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


df = pd.read_pickle(ROOT / "data" / "processed" / "traffic_clean.pkl")
print(f"Loaded {len(df)} hourly rows")


section("1. UNIVARIATE ANALYSIS")
num = ["traffic_volume", "temp_c", "rain_1h", "clouds_all"]
print(df[num].agg(["mean", "median", "std", "min", "max", "skew"]).T.round(2).to_string())
print("\nWeather categories (% of hours):")
print((df.weather_main.value_counts(normalize=True) * 100).round(1).to_string())

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
sns.histplot(df.traffic_volume, bins=60, ax=axes[0], color="#3b6ea5")
axes[0].set_title("Traffic volume (veh/h)")
sns.histplot(df.temp_c, bins=60, ax=axes[1], color="#c96f3b")
axes[1].set_title("Temperature (deg C)")
df.weather_main.value_counts().plot.barh(ax=axes[2], color="#7a7a7a")
axes[2].invert_yaxis()
axes[2].set_title("Weather category (hours)")
savefig(fig, "01_univariate.png")


section("2. BIVARIATE ANALYSIS")
corr_cols = ["traffic_volume", "temp_c", "rain_1h", "snow_1h", "clouds_all", "hour",
             "day_of_week", "month"]
corr = df[corr_cols].corr()
print("Pearson correlation with traffic_volume:")
print(corr.traffic_volume.drop("traffic_volume").round(3)
      .sort_values(key=abs, ascending=False).to_string())
print("Weather correlations are weak. Hour is only moderately correlated because its"
      "\nrelationship with traffic is a curve, not a straight line.")

fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlation matrix")
savefig(fig, "02_correlation_heatmap.png")

weather_median = df.groupby("weather_main").traffic_volume.median().sort_values()
print("\nMedian traffic by weather category:")
print(weather_median.round(0).to_string())
print("Fog and mist mostly occur at night, which is why their medians are low.")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sample = df.sample(5000, random_state=1)
sns.scatterplot(data=sample, x="temp_c", y="traffic_volume", s=8, alpha=0.4,
                color="#3b6ea5", ax=axes[0])
axes[0].set_title(f"Traffic vs temperature (r = {corr.loc['temp_c', 'traffic_volume']:.2f},"
                  " 5,000-hour sample)")
sns.boxplot(data=df, x="traffic_volume", y="weather_main", order=weather_median.index,
            ax=axes[1], color="#9fb7d4", fliersize=1)
axes[1].set_title("Traffic volume by weather")
axes[1].set_ylabel("")
savefig(fig, "03_traffic_vs_weather.png")


section("3. TIME-BASED INDEXING")
ts = df.set_index("date_time").sort_index()
print(f"Index type: {type(ts.index).__name__}, {ts.index.min()} to {ts.index.max()}")
print("\nts.loc['2017-07-04'] (first rows):")
print(ts.loc["2017-07-04", ["traffic_volume", "holiday"]].head(4).to_string())


section("4. RESAMPLING AND GROUPING")
daily = ts.traffic_volume.resample("D").mean()
monthly = ts.traffic_volume.resample("MS").mean()
print(f"Hourly rows: {len(ts)} -> daily: {daily.notna().sum()} -> monthly: "
      f"{monthly.notna().sum()} (periods with no data are left empty)")
print("\nFirst daily means:\n" + daily.dropna().head().round(0).to_string())

by_day = ts.groupby("day_name").traffic_volume.mean().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
by_month = ts.groupby("month").traffic_volume.mean()
by_type = ts.groupby("day_type").traffic_volume.agg(["mean", "median", "count"])
print("\nMean traffic by day of week:\n" + by_day.round(0).to_string())
print("\nMean traffic by day type:\n" + by_type.round(0).to_string())
print("\nMean traffic by month:\n" + by_month.round(0).to_string())


section("5. TIME-SERIES PATTERNS")
profile = ts.groupby(["hour", "day_type"]).traffic_volume.mean().unstack()
wd = profile["Weekday"]
am, pm = wd.loc[:11].idxmax(), wd.loc[12:].idxmax()
we_peak = profile["Weekend"].idxmax()
print(f"Weekday peaks: {am}:00 ({wd[am]:.0f} veh/h) and {pm}:00 ({wd[pm]:.0f} veh/h)")
print(f"Weekend peak: {we_peak}:00 ({profile['Weekend'][we_peak]:.0f} veh/h), no morning rush")
print(f"At 7:00 weekday traffic is {wd[7] / profile['Weekend'][7]:.1f}x weekend traffic")
print(f"Average weekday traffic is "
      f"{by_type.loc['Weekday', 'mean'] / by_type.loc['Weekend', 'mean'] - 1:.0%} higher than "
      f"weekend; holidays ({by_type.loc['Holiday', 'mean']:.0f}) are close to weekends "
      f"({by_type.loc['Weekend', 'mean']:.0f})")
print(f"Lowest month: {by_month.idxmin()} ({by_month.min():.0f}), highest: {by_month.idxmax()} "
      f"({by_month.max():.0f}), a {by_month.max() / by_month.min() - 1:.0%} difference")
# shift by one clock hour, not one row, so hours across a gap are not paired
lag1 = ts.traffic_volume.corr(ts.traffic_volume.shift(1, freq="h"))
print(f"Correlation with the previous hour: {lag1:.2f}")

fig, ax = plt.subplots(figsize=(10, 5))
for dt, col in DAY_COLORS.items():
    ax.plot(profile.index, profile[dt], marker="o", ms=3, color=col, label=dt)
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("hour of day")
ax.set_ylabel("mean traffic volume (veh/h)")
ax.set_title("Average hourly traffic profile")
ax.legend()
savefig(fig, "04_hourly_profile.png")

hw = ts.pivot_table(index="day_name", columns="hour", values="traffic_volume",
                    aggfunc="mean").reindex(by_day.index)
fig, ax = plt.subplots(figsize=(14, 4.2))
sns.heatmap(hw, cmap="YlOrRd", ax=ax, cbar_kws={"label": "mean veh/h"})
ax.set_title("Mean traffic: day of week x hour")
ax.set_ylabel("")
savefig(fig, "05_heatmap_day_hour.png")

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
axes[0].plot(daily.index, daily, lw=0.5, color="#9fb7d4", label="daily mean")
axes[0].plot(daily.index, daily.rolling(30, min_periods=15).mean(), color="#b5332e",
             label="30-day rolling mean")
axes[0].set_title("Daily mean traffic (gap = sensor outage)")
axes[0].legend(loc="lower left")
axes[1].plot(monthly.index, monthly, marker="o", ms=3, color="#c96f3b")
axes[1].set_title("Monthly mean traffic")
for ax in axes:
    ax.set_ylabel("veh/h")
savefig(fig, "06_daily_monthly_series.png")

fig, ax = plt.subplots(figsize=(9, 4.5))
by_month.plot.bar(ax=ax, color="#3b6ea5")
ax.set_ylim(by_month.min() * 0.9, by_month.max() * 1.03)
ax.set_xlabel("month")
ax.set_ylabel("mean veh/h")
ax.set_title("Mean traffic by month")
plt.xticks(rotation=0)
savefig(fig, "07_monthly_seasonality.png")
