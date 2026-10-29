"""Phase 6 - Linear Regression and Model Evaluation (Module 07)."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase6"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def evaluate(y, pred):
    return {"MAE": mean_absolute_error(y, pred),
            "RMSE": np.sqrt(mean_squared_error(y, pred)),
            "R2": r2_score(y, pred)}


section("1. FEATURES AND TARGET")
df = pd.read_pickle(ROOT / "data" / "processed" / "traffic_clean.pkl")
ts = df.set_index("date_time").sort_index()
# full hourly index so that shift(1) is exactly one hour back
ts = ts.reindex(pd.date_range(ts.index.min(), ts.index.max(), freq="h"))
vol = ts.traffic_volume

# target is the volume at t+1; all features are known at time t
m = pd.DataFrame(index=ts.index)
m["target_next_hour"] = vol.shift(-1)
m["vol_now"] = vol
m["vol_prev_hour"] = vol.shift(1)
m["vol_same_hour_yesterday"] = vol.shift(23)
m["vol_same_hour_last_week"] = vol.shift(167)
m["temp_c"] = ts.temp_c
m["rain_1h"] = ts.rain_1h
m["snow_1h"] = ts.snow_1h
m["clouds_all"] = ts.clouds_all
# calendar of the next hour is known in advance
next_day_type = ts.day_type.shift(-1)
m["next_is_weekend"] = (next_day_type == "Weekend").astype(float)
m["next_is_holiday"] = (next_day_type == "Holiday").astype(float)
m.loc[next_day_type.isna(), ["next_is_weekend", "next_is_holiday"]] = np.nan
next_hour = pd.Series((m.index + pd.Timedelta(hours=1)).hour, index=m.index)
hour_dummies = pd.get_dummies(next_hour, prefix="next_hour", drop_first=True, dtype=float)
m = pd.concat([m, hour_dummies], axis=1).dropna()

HISTORY = ["vol_now", "vol_prev_hour", "vol_same_hour_yesterday", "vol_same_hour_last_week"]
WEATHER = ["temp_c", "rain_1h", "snow_1h", "clouds_all"]
CALENDAR = ["next_is_weekend", "next_is_holiday"] + list(hour_dummies.columns)
print(f"Rows: {len(m)}")
print(f"Features: {len(HISTORY)} traffic history, {len(WEATHER)} weather, "
      f"{len(CALENDAR)} calendar (weekend, holiday, hour dummies)")


section("2. TRAIN / TEST SPLIT")
SPLIT = pd.Timestamp("2017-10-01")
train, test = m[m.index < SPLIT], m[m.index >= SPLIT]
print(f"Train: {train.index.min():%Y-%m-%d} to {train.index.max():%Y-%m-%d} ({len(train)} hours)")
print(f"Test : {test.index.min():%Y-%m-%d} to {test.index.max():%Y-%m-%d} ({len(test)} hours)")
print("Split by time, not randomly, so the model never trains on future data.")
y_tr, y_te = train.target_next_hour, test.target_next_hour


section("3. MODELS AND BASELINES")
preds = {
    "Baseline: next hour = this hour": test.vol_now,
    "Baseline: same hour last week": test.vol_same_hour_last_week,
}
models = {}
for name, cols in [("LinReg A: calendar + weather", CALENDAR + WEATHER),
                   ("LinReg B: + traffic history", CALENDAR + WEATHER + HISTORY)]:
    lr = LinearRegression().fit(train[cols], y_tr)
    models[name] = (lr, cols)
    preds[name] = pd.Series(lr.predict(test[cols]), index=test.index)

res = pd.DataFrame({k: evaluate(y_te, p) for k, p in preds.items()}).T
print("Test set (Oct 2017 - Sep 2018):")
print(res.round(3).to_string())

best = "LinReg B: + traffic history"
lr, cols = models[best]
tr = evaluate(y_tr, lr.predict(train[cols]))
print(f"\nModel B on the training set: MAE {tr['MAE']:.0f}, RMSE {tr['RMSE']:.0f}, R2 {tr['R2']:.3f}")
imp = 1 - res.loc[best, "MAE"] / res.loc["Baseline: next hour = this hour", "MAE"]
print(f"MAE reduction vs 'next hour = this hour': {imp:.0%}")

coef = pd.Series(lr.coef_, index=cols)
print("\nCoefficients (hour dummies not shown):")
print(coef.drop(hour_dummies.columns).round(3).to_string())


section("4. ERROR ANALYSIS")
err = pd.DataFrame({"actual": y_te, "pred": preds[best]})
err["error"] = err.pred - err.actual
err["abs_error"] = err.error.abs()
err["hour"] = (err.index + pd.Timedelta(hours=1)).hour
err["day_type"] = ts.day_type.shift(-1).reindex(err.index)

by_hour = err.groupby("hour").abs_error.mean()
print("Hours with the highest MAE:")
print(by_hour.sort_values(ascending=False).head(5).round(0).to_string())
print("\nMAE by day type:")
print(err.groupby("day_type").abs_error.agg(["mean", "count"]).round(0).to_string())
print(f"\nMean error: {err.error.mean():+.1f} veh/h")

clusters = pd.read_csv(ROOT / "data" / "processed" / "daily_clusters.csv", index_col=0,
                       parse_dates=True)
worst_days = err.groupby(err.index.normalize()).abs_error.mean().nlargest(5)
print("\nDays with the largest MAE:")
for d, e in worst_days.items():
    hol = ts.holiday.get(d, "None")
    snow = (ts.loc[d:d + pd.Timedelta(hours=23), "weather_main"] == "Snow").mean()
    if d in clusters.index:
        c = clusters.loc[d]
        phase4 = c.cluster + (" + outlier" if c.outlier else "")
    else:
        phase4 = "not clustered"
    print(f"  {d:%Y-%m-%d} {d.day_name():<9} MAE {e:4.0f}  holiday: {hol:<17} "
          f"snowy: {snow:4.0%}  phase 4: {phase4}")

fig, ax = plt.subplots(figsize=(7, 6.5))
ax.scatter(err.actual, err.pred, s=4, alpha=0.3, color="#3b6ea5")
lim = [0, max(err.actual.max(), err.pred.max())]
ax.plot(lim, lim, color="#b5332e", lw=1.5, label="perfect prediction")
ax.set_xlabel("actual traffic volume (veh/h)")
ax.set_ylabel("predicted traffic volume (veh/h)")
ax.set_title(f"Actual vs predicted (test set)\nMAE {res.loc[best, 'MAE']:.0f}, "
             f"RMSE {res.loc[best, 'RMSE']:.0f}, R2 {res.loc[best, 'R2']:.3f}")
ax.legend()
savefig(fig, "01_actual_vs_predicted.png")

week = err.loc["2018-03-04":"2018-03-11"]
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(week.index, week.actual, color="black", lw=1.5, label="actual")
ax.plot(week.index, week.pred, color="#c96f3b", lw=1.5, ls="--", label="predicted")
ax.set_ylabel("veh/h")
ax.set_title("Actual vs predicted, 2018-03-04 to 2018-03-11")
ax.legend()
savefig(fig, "02_test_week.png")

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
by_hour.plot.bar(ax=axes[0], color="#3b6ea5")
axes[0].set_xlabel("hour being predicted")
axes[0].set_ylabel("mean absolute error (veh/h)")
axes[0].set_title("Error by hour of day")
axes[0].tick_params(axis="x", rotation=0)
sns.histplot(err.error, bins=80, ax=axes[1], color="#9fb7d4")
axes[1].axvline(0, color="#b5332e")
axes[1].set_xlim(-2500, 2500)
axes[1].set_xlabel("prediction error (veh/h)")
axes[1].set_title("Residuals")
savefig(fig, "03_error_analysis.png")

fig, ax = plt.subplots(figsize=(10, 4.5))
res[["MAE", "RMSE"]].plot.barh(ax=ax, color=["#3b6ea5", "#c96f3b"])
ax.invert_yaxis()
ax.set_xlabel("veh/h")
ax.set_title("Model vs baselines (test set)")
savefig(fig, "04_model_comparison.png")
