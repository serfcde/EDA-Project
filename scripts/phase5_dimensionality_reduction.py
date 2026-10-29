"""Phase 5 - Dimensionality Reduction: PCA and SVD (Module 06)."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase5"
OUT.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


section("1. FEATURE MATRIX")
df = pd.read_pickle(ROOT / "data" / "processed" / "traffic_clean.pkl")
F = pd.DataFrame({
    "traffic_volume": df.traffic_volume,
    "temp_c": df.temp_c,
    "rain_1h": df.rain_1h,
    "snow_1h": df.snow_1h,
    "clouds_all": df.clouds_all,
    # hour is cyclic, so 23:00 and 00:00 should be close together
    "hour_sin": np.sin(2 * np.pi * df.hour / 24),
    "hour_cos": np.cos(2 * np.pi * df.hour / 24),
    "is_weekend": (df.day_type != "Weekday").astype(int),
})
print(f"Matrix: {F.shape[0]} hours x {F.shape[1]} features")
print("Features: " + ", ".join(F.columns))

Z = StandardScaler().fit_transform(F)
print("\nAfter standardizing:")
print(pd.DataFrame(Z, columns=F.columns).agg(["mean", "std"]).round(2).to_string())


section("2. PCA")
pca = PCA().fit(Z)
evr = pd.Series(pca.explained_variance_ratio_, index=[f"PC{i + 1}" for i in range(Z.shape[1])])
table = pd.DataFrame({"eigenvalue": pca.explained_variance_, "% variance": evr * 100,
                      "cumulative %": evr.cumsum() * 100})
print(table.round(2).to_string())
k80 = int((evr.cumsum() >= 0.80).argmax() + 1)
print(f"\nComponents for 80% of the variance: {k80} of {Z.shape[1]}")
print(f"Components with eigenvalue > 1 (Kaiser rule): {int((pca.explained_variance_ > 1).sum())}")

load = pd.DataFrame(pca.components_[:4].T, index=F.columns, columns=["PC1", "PC2", "PC3", "PC4"])
print("\nLoadings:")
print(load.round(2).to_string())
for pc in load.columns:
    top = load[pc].abs().sort_values(ascending=False).index[:3]
    print(f"  {pc}: " + ", ".join(f"{f} ({load.loc[f, pc]:+.2f})" for f in top))
print("\nPC1: traffic volume and time of day (busy daytime vs quiet night)"
      "\nPC2: temperature vs cloud cover"
      "\nPC3: rain and cloud cover"
      "\nPC4: snow and weekend"
      "\nTraffic volume loads almost only on PC1, so weather shares little variance with it.")

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(evr.index, evr * 100, color="#9fb7d4", label="individual")
ax.plot(evr.index, evr.cumsum() * 100, marker="o", color="#b5332e", label="cumulative")
ax.axhline(80, color="grey", ls="--", lw=1)
ax.text(len(evr) - 1, 81.5, "80%", ha="right", color="grey")
ax.set_ylabel("% of variance explained")
ax.set_title("PCA explained variance")
ax.legend(loc="center right")
savefig(fig, "01_explained_variance.png")

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(load, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
ax.set_title("PCA loadings (first 4 components)")
savefig(fig, "02_loadings.png")

scores = pca.transform(Z)
s = np.random.default_rng(0).choice(len(df), 6000, replace=False)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
LEVEL_COLORS = {"Low": "#f2d16b", "Medium": "#e8893a", "High": "#b5332e"}
for lvl, col in LEVEL_COLORS.items():
    m = df.traffic_level.values[s] == lvl
    axes[0].scatter(scores[s][m, 0], scores[s][m, 1], s=6, alpha=0.5, color=col, label=lvl)
axes[0].set_title("PC1 vs PC2, coloured by traffic level")
axes[0].legend(title="traffic level", markerscale=3)
sc = axes[1].scatter(scores[s, 0], scores[s, 1], c=df.hour.values[s], cmap="twilight",
                     s=6, alpha=0.6)
fig.colorbar(sc, ax=axes[1], label="hour of day")
axes[1].set_title("PC1 vs PC2, coloured by hour")
for ax in axes:
    ax.set_xlabel(f"PC1 ({evr['PC1']:.0%})")
    ax.set_ylabel(f"PC2 ({evr['PC2']:.0%})")
savefig(fig, "03_pca_2d.png")

lvl_means = pd.DataFrame(scores[:, :2], columns=["PC1", "PC2"]).groupby(
    df.traffic_level.values, observed=True).mean().reindex(["Low", "Medium", "High"])
print("\nMean PC scores per traffic level:")
print(lvl_means.round(2).to_string())


section("3. SVD")
U, S, Vt = np.linalg.svd(Z, full_matrices=False)
n = Z.shape[0]
print(f"Z = U S V^T with U {U.shape}, S {S.shape}, V^T {Vt.shape}")
print("Singular values: " + ", ".join(f"{v:.1f}" for v in S))
print(f"Rank: {np.linalg.matrix_rank(Z)}")
print(f"Condition number: {S[0] / S[-1]:.2f}")

eig_from_svd = S ** 2 / (n - 1)
cmp = pd.DataFrame({"PCA eigenvalue": pca.explained_variance_,
                    "S^2 / (n-1)": eig_from_svd}, index=evr.index)
print("\nPCA eigenvalues vs squared singular values:")
print(cmp.round(4).to_string())
print(f"\nRows of V^T equal PCA components (up to sign): "
      f"{np.allclose(np.abs(Vt), np.abs(pca.components_), atol=1e-6)}")
print(f"U * S equals PCA scores (up to sign): "
      f"{np.allclose(np.abs(U * S), np.abs(scores), atol=1e-6)}")

rows = []
for k in range(1, Z.shape[1] + 1):
    Zk = (U[:, :k] * S[:k]) @ Vt[:k]
    rows.append({"k": k, "relative error": np.linalg.norm(Z - Zk) / np.linalg.norm(Z),
                 "variance kept %": 100 * (S[:k] ** 2).sum() / (S ** 2).sum()})
print("\nRank-k reconstruction of Z:")
print(pd.DataFrame(rows).set_index("k").round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
axes[0].bar(range(1, len(S) + 1), S, color="#3b6ea5")
axes[0].set_xlabel("index")
axes[0].set_ylabel("singular value")
axes[0].set_title("Singular values")
axes[1].plot(evr.index, pca.explained_variance_, marker="o", ms=10, label="PCA eigenvalues",
             color="#9fb7d4")
axes[1].plot(evr.index, eig_from_svd, marker="x", ms=8, ls="--", color="#b5332e",
             label="S^2 / (n-1) from SVD")
axes[1].set_title("PCA eigenvalues vs SVD")
axes[1].legend()
savefig(fig, "04_svd_vs_pca.png")
