"""Phase 4 - Clustering (GMM / EM) and Outlier Detection (Module 05)."""
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase4"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


section("1. DAY-LEVEL FEATURES")
df = pd.read_pickle(ROOT / "data" / "processed" / "traffic_clean.pkl")
df["date"] = df.date_time.dt.normalize()
hours_per_day = df.groupby("date").size()
d = df[df.date.isin(hours_per_day.index[hours_per_day >= 20])]

PERIODS = ["Night (0-5)", "AM peak (6-9)", "Midday (10-15)", "PM peak (16-19)",
           "Evening (20-23)"]
X = d.pivot_table(index="date", columns="time_of_day", values="traffic_volume",
                  aggfunc="mean", observed=True)[PERIODS].dropna()
info = d.groupby("date").agg(day_name=("day_name", "first"), day_type=("day_type", "first"),
                             holiday=("holiday", "first"),
                             snowy_hours=("weather_main", lambda s: (s == "Snow").mean()),
                             daily_mean=("traffic_volume", "mean")).loc[X.index]
print(f"Days with at least 20 recorded hours: {len(X)}")
print("Mean volume per period of the day:\n" + X.head().round(0).to_string())
Xs = StandardScaler().fit_transform(X)


section("2. GAUSSIAN MIXTURE MODEL (EM)")
bic = pd.Series({k: GaussianMixture(k, covariance_type="full", n_init=5, random_state=0)
                 .fit(Xs).bic(Xs) for k in range(1, 7)}, name="BIC")
print("BIC by number of clusters:\n" + bic.round(0).to_string())

# K=4 also lowers BIC, but only by splitting normal weekdays into early/late week
lab4 = GaussianMixture(4, covariance_type="full", n_init=10, random_state=0).fit(Xs).predict(Xs)
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
ct4 = pd.crosstab(lab4, info.day_name)[weekdays]
split = ct4[ct4.sum(axis=1) > 400]
print("\nWith K = 4 the two largest weekday clusters are:")
print(split.to_string())
print("Evening mean of these two: " + ", ".join(
    f"{X.loc[lab4 == c, 'Evening (20-23)'].mean():.0f}" for c in split.index) + " veh/h")
K = 3
print(f"The extra cluster only separates early-week from late-week weekdays, so K = {K} is used.")

gmm = GaussianMixture(K, covariance_type="full", n_init=10, random_state=0).fit(Xs)
print(f"\nEM converged: {gmm.converged_} after {gmm.n_iter_} iterations")
lab = gmm.predict(Xs)
prob = gmm.predict_proba(Xs).max(axis=1)

means = pd.Series(info.daily_mean.values).groupby(lab).mean().sort_values()
names = dict(zip(means.index, ["Weekend / holiday", "Reduced weekday", "Normal weekday"]))
info["cluster"] = [names[c] for c in lab]
info["membership_prob"] = prob
ORDER = ["Normal weekday", "Reduced weekday", "Weekend / holiday"]
COLORS = {"Normal weekday": "#3b6ea5", "Reduced weekday": "#8e5ea2",
          "Weekend / holiday": "#4c9a5f"}

prof = X.groupby(info.cluster.values).mean().loc[ORDER]
print("\nCluster profiles (mean veh/h):")
print(prof.round(0).to_string())
print("\nClusters vs day type (days):")
print(pd.crosstab(info.cluster, info.day_type).loc[ORDER]
      .assign(total=lambda t: t.sum(axis=1)).to_string())

red = info[info.cluster == "Reduced weekday"]
print("\nReduced weekday cluster by day of week:")
print(red.day_name.value_counts().to_string())
print(f"Snowy hours: {red.snowy_hours.mean():.0%} in this cluster, "
      f"{info.snowy_hours.mean():.0%} over all days")
print("Most common months: " + ", ".join(
    str(m) for m in red.index.month.value_counts().index[:3]))
print(f"\nMean membership probability: {prob.mean():.2f} "
      f"({(prob < 0.8).sum()} days below 0.8)")

fig, ax = plt.subplots(figsize=(10, 5))
for c in ORDER:
    ax.plot(PERIODS, prof.loc[c], marker="o", lw=2, color=COLORS[c],
            label=f"{c} (n={(info.cluster == c).sum()})")
ax.set_ylabel("mean traffic volume (veh/h)")
ax.set_title("GMM cluster profiles")
ax.legend()
savefig(fig, "01_cluster_profiles.png")


section("3. OUTLIER DETECTION")
info["log_likelihood"] = gmm.score_samples(Xs)
info["outlier"] = info.log_likelihood < info.log_likelihood.quantile(0.01)
out = info[info.outlier].sort_values("log_likelihood")
print(f"Outliers = lowest 1% log-likelihood under the fitted GMM ({len(out)} days):")
print(out[["day_name", "day_type", "holiday", "cluster", "snowy_hours", "daily_mean"]]
      .round(2).to_string())
print(f"\nMostly snowy (>= 50% of hours): {(out.snowy_hours >= 0.5).sum()} days")
print(f"Very low traffic (< 1500 veh/h average): {(out.daily_mean < 1500).sum()} days")

fig, ax = plt.subplots(figsize=(9, 6))
for c in ORDER:
    m = info.cluster == c
    ax.scatter(X.loc[m, "AM peak (6-9)"], X.loc[m, "Midday (10-15)"], s=10, alpha=0.6,
               color=COLORS[c], label=c)
ax.scatter(X.loc[info.outlier, "AM peak (6-9)"], X.loc[info.outlier, "Midday (10-15)"],
           s=80, facecolors="none", edgecolors="black", label="outlier")
ax.set_xlabel("AM peak mean traffic (veh/h)")
ax.set_ylabel("midday mean traffic (veh/h)")
ax.set_title("Clusters and outliers")
ax.legend()
savefig(fig, "02_clusters_outliers_scatter.png")

info.to_csv(ROOT / "data" / "processed" / "daily_clusters.csv")
print("\nSaved data/processed/daily_clusters.csv")
