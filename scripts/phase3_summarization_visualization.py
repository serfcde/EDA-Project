"""Phase 3 - Data Summarization and Visualization (Module 04)."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase3"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
LEVELS = ["Low", "Medium", "High"]
LEVEL_COLORS = ["#f2d16b", "#e8893a", "#b5332e"]


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


df = pd.read_pickle(ROOT / "data" / "processed" / "traffic_clean.pkl")


section("1. STATISTICAL SUMMARY MEASURES")
num = ["traffic_volume", "temp_c", "rain_1h", "clouds_all"]
q1, q3 = df[num].quantile(0.25), df[num].quantile(0.75)
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
summary = pd.DataFrame({
    "mean": df[num].mean(), "median": df[num].median(), "std": df[num].std(),
    "min": df[num].min(), "Q1": q1, "Q3": q3, "max": df[num].max(), "IQR": iqr,
    "outliers": ((df[num] < lo) | (df[num] > hi)).sum(),
})
print(summary.round(2).to_string())
print("\nOutliers use the 1.5 x IQR rule. rain_1h has many because almost every hour"
      "\nhas 0 rain, so any rainy hour falls outside the range.")

print("\nTraffic volume by day type:")
print(df.groupby("day_type").traffic_volume.describe().round(0).to_string())

# same IQR rule, but within each hour and day type
g = df.groupby(["hour", "day_type"]).traffic_volume
gq1, gq3 = g.transform(lambda s: s.quantile(0.25)), g.transform(lambda s: s.quantile(0.75))
unusual = (df.traffic_volume < gq1 - 1.5 * (gq3 - gq1)) | \
          (df.traffic_volume > gq3 + 1.5 * (gq3 - gq1))
print(f"\nHours unusual for their hour and day type: {unusual.sum()} ({unusual.mean():.1%})")

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="day_type", y="traffic_volume", order=["Weekday", "Weekend", "Holiday"],
            color="#9fb7d4", ax=ax)
ax.set_title("Traffic volume by day type")
ax.set_xlabel("")
ax.set_ylabel("veh/h")
savefig(fig, "01_boxplot_day_type.png")


section("2. CONTINGENCY TABLES")


def cramers_v(ct):
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    print(f"Chi-square = {chi2:.1f}, dof = {dof}, p = {p:.2e}, Cramer's V = {v:.3f}")
    return v


def contingency(col, title):
    print(f"\n{title}:")
    print(pd.crosstab(df[col], df.traffic_level, margins=True).to_string())
    print("Row %:")
    print((pd.crosstab(df[col], df.traffic_level, normalize="index") * 100).round(1)
          .to_string())
    return cramers_v(pd.crosstab(df[col], df.traffic_level)[LEVELS])


v_hol = contingency("is_holiday", "Holiday status x traffic level")
v_day = contingency("day_type", "Day type x traffic level")

main_weather = df.weather_main.value_counts().index[:6]
sub = df[df.weather_main.isin(main_weather)]
ct_w = pd.crosstab(sub.weather_main, sub.traffic_level)[LEVELS]
print("\nWeather (6 most common) x traffic level, row %:")
print((ct_w.div(ct_w.sum(axis=1), axis=0) * 100).round(1).to_string())
v_w = cramers_v(ct_w)
print(f"\nDay type (V = {v_day:.2f}) is more strongly associated with traffic level than"
      f" weather (V = {v_w:.2f}).")

fig, ax = plt.subplots(figsize=(7, 3.5))
sns.heatmap(pd.crosstab(df.day_type, df.traffic_level, normalize="index")[LEVELS] * 100,
            annot=True, fmt=".0f", cmap="YlOrRd", ax=ax, cbar_kws={"label": "row %"})
ax.set_title("Day type x traffic level (row %)")
ax.set_ylabel("")
savefig(fig, "02_contingency_day_type.png")


section("3. PLOTS")
s = df.sample(5000, random_state=1)
# small horizontal jitter so points at the same hour don't overlap
jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(s))
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=s, x=s.hour + jitter, y="traffic_volume", hue="traffic_level",
                hue_order=LEVELS, palette=LEVEL_COLORS, s=10, alpha=0.6, edgecolor=None, ax=ax)
ax.set_xlabel("hour of day")
ax.set_ylabel("traffic volume (veh/h)")
ax.set_xticks(range(0, 24, 2))
ax.set_title("Traffic volume by hour (5,000-hour sample)")
savefig(fig, "03_scatter_hour_volume.png")

DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dot = df[~df.is_holiday].groupby(["day_name", "time_of_day"], observed=True) \
    .traffic_volume.median().unstack().reindex(DOW)
fig, ax = plt.subplots(figsize=(9, 5))
for col, c in zip(dot.columns, sns.color_palette("viridis", len(dot.columns))):
    ax.plot(dot[col], dot.index, "o", ms=9, label=col, color=c)
ax.invert_yaxis()
ax.set_xlabel("median traffic volume (veh/h)")
ax.set_title("Median traffic by day and time of day")
ax.legend(fontsize=8, loc="lower right")
savefig(fig, "04_dot_chart.png")

ct_t = pd.crosstab(df.time_of_day, df.traffic_level, normalize="index")[LEVELS] * 100
fig, ax = plt.subplots(figsize=(10, 5))
ct_t.plot.bar(stacked=True, ax=ax, color=LEVEL_COLORS, width=0.7)
ax.set_ylabel("% of hours")
ax.set_xlabel("")
ax.set_title("Traffic level by time of day")
ax.legend(title="traffic level", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.xticks(rotation=0)
savefig(fig, "05_bar_time_of_day.png")
print("Traffic level by time of day (row %):\n" + ct_t.round(1).to_string())

hi_share = pd.crosstab(df.day_type, df.traffic_level, normalize="index")["High"] * 100
print(f"\n'High' traffic hours: weekday {hi_share['Weekday']:.0f}%, weekend "
      f"{hi_share['Weekend']:.0f}%, holiday {hi_share['Holiday']:.0f}%")
