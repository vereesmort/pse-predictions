"""
01_data_exploration.py
======================
Comprehensive data exploration of the Decagon polypharmacy dataset.

Outputs
-------
- results/plots/  : all figures
- results/metrics/exploration_summary.csv
- results/metrics/per_se_missing_targets.csv

Run:
    python 01_data_exploration.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

from configs.config import (
    RAW_FILES, OUTPUT_DIR, PLOTS_DIR, METRICS_DIR,
    MIN_PAIRS_PER_SE, REPRESENTATIVE_SES, RANDOM_SEED,
)

os.makedirs(PLOTS_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ── 1. Load raw data ───────────────────────────────────────────────────────────
print("Loading raw data...")
combo    = pd.read_csv(RAW_FILES["combo"])
mono     = pd.read_csv(RAW_FILES["mono"])
ppi      = pd.read_csv(RAW_FILES["ppi"])
targets  = pd.read_csv(RAW_FILES["targets"])

# ── 2. Filter combo to 963 SEs with >= 500 pairs ──────────────────────────────
se_counts     = combo.groupby("Polypharmacy Side Effect").size()
se_500        = se_counts[se_counts >= MIN_PAIRS_PER_SE].index
combo_f       = combo[combo["Polypharmacy Side Effect"].isin(se_500)].copy()
all_drugs     = pd.unique(combo_f[["STITCH 1", "STITCH 2"]].values.ravel())
t_drugs       = set(targets["STITCH"])
ppi_genes     = set(ppi["Gene 1"]).union(set(ppi["Gene 2"]))
target_genes  = set(targets["Gene"])

# ── 3. High-level statistics ───────────────────────────────────────────────────
print("\n=== Dataset Statistics ===")
stats = {
    "total_combo_rows"              : len(combo),
    "filtered_combo_rows"           : len(combo_f),
    "total_unique_se"               : combo["Polypharmacy Side Effect"].nunique(),
    "filtered_unique_se"            : len(se_500),
    "unique_drugs_in_filtered_combo": len(all_drugs),
    "drugs_with_targets"            : len(t_drugs & set(all_drugs)),
    "drugs_without_targets"         : len(set(all_drugs) - t_drugs),
    "pct_drugs_missing_targets"     : round(100 * len(set(all_drugs) - t_drugs) / len(all_drugs), 2),
    "ppi_nodes"                     : len(ppi_genes),
    "ppi_edges"                     : len(ppi),
    "target_genes"                  : len(target_genes),
    "target_genes_in_ppi"           : len(target_genes & ppi_genes),
    "mono_drugs"                    : mono["STITCH"].nunique(),
    "mono_unique_se"                : mono["Individual Side Effect"].nunique(),
}
for k, v in stats.items():
    print(f"  {k:45s}: {v}")

# ── 4. Per-SE missing target analysis (vectorised) ─────────────────────────────
print("\nComputing per-SE missing target coverage...")
combo_f = combo_f.copy()
combo_f["d1_missing"] = ~combo_f["STITCH 1"].isin(t_drugs)
combo_f["d2_missing"] = ~combo_f["STITCH 2"].isin(t_drugs)
combo_f["both_missing"] = combo_f["d1_missing"] & combo_f["d2_missing"]

se_names_map = combo_f.groupby("Polypharmacy Side Effect")["Side Effect Name"].first()

def per_se_stats(grp):
    se = grp.name
    drugs = pd.unique(grp[["STITCH 1", "STITCH 2"]].values.ravel())
    n_total   = len(drugs)
    n_missing = sum(1 for d in drugs if d not in t_drugs)
    n_pairs   = len(grp)
    both_miss = int(grp["both_missing"].sum())
    return pd.Series({
        "Side Effect Name"               : se_names_map[se],
        "Total Unique Drugs"             : n_total,
        "Drugs Without Target"           : n_missing,
        "Pct Drugs Without Target"       : round(100 * n_missing / n_total, 2),
        "Total Pairs"                    : n_pairs,
        "Pairs Both Drugs Missing Target": both_miss,
        "Pct Pairs Both Missing"         : round(100 * both_miss / n_pairs, 2),
    })

se_df = combo_f.groupby("Polypharmacy Side Effect").apply(per_se_stats)
se_df.index.name = "Polypharmacy Side Effect"
se_df = se_df.reset_index()

# Flag representative SEs
se_df["is_representative"] = se_df["Polypharmacy Side Effect"].isin(REPRESENTATIVE_SES)
se_df.to_csv(os.path.join(METRICS_DIR, "per_se_missing_targets.csv"), index=False)
print(f"  Saved per_se_missing_targets.csv  ({len(se_df)} rows)")

# ── 5. Summary statistics ──────────────────────────────────────────────────────
print("\n=== Per-SE Missing Target Summary ===")
desc = se_df["Pct Drugs Without Target"].describe(percentiles=[.1,.25,.5,.75,.9])
print(desc.round(2))

# ── 6. PPI degree distribution ─────────────────────────────────────────────────
deg_counter = Counter(ppi["Gene 1"].tolist() + ppi["Gene 2"].tolist())
deg_vals = np.array(list(deg_counter.values()))
print(f"\nPPI degree: mean={deg_vals.mean():.1f}, median={np.median(deg_vals):.0f}, max={deg_vals.max()}")

# ── 7. Drug pair SE count distribution ────────────────────────────────────────
pair_se_cnt = combo_f.groupby(["STITCH 1", "STITCH 2"])["Polypharmacy Side Effect"].count()
print(f"\nSEs per drug pair: mean={pair_se_cnt.mean():.1f}, max={pair_se_cnt.max()}, min={pair_se_cnt.min()}")

# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

# Figure 1 — SE sample size distribution (log scale) with representative SEs marked
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(se_df["Total Pairs"], bins=60, color="#378ADD", edgecolor="white", linewidth=0.4)
rep_df = se_df[se_df["is_representative"]]
for _, row in rep_df.iterrows():
    ax.axvline(row["Total Pairs"], color="#D85A30", alpha=0.6, linewidth=0.8, linestyle="--")
ax.set_xlabel("Drug pairs per side effect")
ax.set_ylabel("Number of side effects")
ax.set_title("Distribution of sample sizes (963 SEs)")
ax.set_yscale("log")

ax = axes[1]
ax.hist(se_df["Pct Drugs Without Target"], bins=40, color="#1D9E75", edgecolor="white", linewidth=0.4)
ax.axvline(se_df["Pct Drugs Without Target"].mean(), color="#D85A30", linewidth=1.5,
           linestyle="--", label=f"Mean: {se_df['Pct Drugs Without Target'].mean():.1f}%")
ax.set_xlabel("% drugs without protein target")
ax.set_ylabel("Number of side effects")
ax.set_title("Missing target coverage per SE")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig1_se_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved fig1_se_distributions.png")

# Figure 2 — Scatter: sample size vs missing target rate, coloured by bin
fig, ax = plt.subplots(figsize=(10, 6))

def size_bin(p):
    if p > 6594: return "high", "#378ADD"
    if p > 2970: return "medium", "#1D9E75"
    if p > 1231: return "low", "#EF9F27"
    return "very_low", "#E24B4A"

colors = [size_bin(p)[1] for p in se_df["Total Pairs"]]
ax.scatter(se_df["Total Pairs"], se_df["Pct Drugs Without Target"],
           c=colors, alpha=0.5, s=18, linewidths=0)

# Highlight representative SEs
for _, row in rep_df.iterrows():
    ax.scatter(row["Total Pairs"], row["Pct Drugs Without Target"],
               s=120, zorder=5, edgecolors="black", linewidths=0.8,
               color=size_bin(row["Total Pairs"])[1])
    ax.annotate(row["Side Effect Name"][:22],
                (row["Total Pairs"], row["Pct Drugs Without Target"]),
                fontsize=6.5, xytext=(5, 3), textcoords="offset points")

from matplotlib.patches import Patch
legend_els = [Patch(color=c, label=l) for l, c in
              [("High pairs (>6594)", "#378ADD"), ("Medium pairs", "#1D9E75"),
               ("Low pairs", "#EF9F27"), ("Very low pairs (<1231)", "#E24B4A")]]
ax.legend(handles=legend_els, fontsize=8)
ax.set_xlabel("Total drug pairs (log scale)")
ax.set_xscale("log")
ax.set_ylabel("% drugs without protein target")
ax.set_title("Sample size vs. missing target rate — 963 side effects\n(outlined = representative 12 SEs)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig2_scatter_size_vs_missing.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_scatter_size_vs_missing.png")

# Figure 3 — PPI degree distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(deg_vals, bins=80, color="#7F77DD", edgecolor="white", linewidth=0.3, log=True)
ax.set_xlabel("Node degree in PPI")
ax.set_ylabel("Count (log scale)")
ax.set_title("PPI protein degree distribution")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig3_ppi_degree.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3_ppi_degree.png")

# Figure 4 — SEs per drug pair distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(pair_se_cnt.values, bins=60, color="#D4537E", edgecolor="white", linewidth=0.3, log=True)
ax.set_xlabel("Number of side effects per drug pair")
ax.set_ylabel("Count (log scale)")
ax.set_title("Side effects per drug pair distribution")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig4_se_per_pair.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig4_se_per_pair.png")

# Figure 5 — 12 representative SEs: detailed bar chart
rep_sorted = rep_df.sort_values("Total Pairs", ascending=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
bars = ax.barh(rep_sorted["Side Effect Name"], rep_sorted["Total Pairs"],
               color="#378ADD", alpha=0.85)
ax.set_xlabel("Total drug pairs")
ax.set_title("Representative SEs — sample size")

ax = axes[1]
colors_miss = []
for pct in rep_sorted["Pct Drugs Without Target"]:
    if pct > 55: colors_miss.append("#E24B4A")
    elif pct > 41: colors_miss.append("#EF9F27")
    else: colors_miss.append("#1D9E75")
ax.barh(rep_sorted["Side Effect Name"], rep_sorted["Pct Drugs Without Target"],
        color=colors_miss, alpha=0.85)
ax.axvline(55, color="#E24B4A", linestyle="--", linewidth=1, label="55% threshold")
ax.axvline(41, color="#1D9E75", linestyle="--", linewidth=1, label="41% threshold")
ax.set_xlabel("% drugs without protein target")
ax.set_title("Representative SEs — missing target rate")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fig5_representative_ses.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig5_representative_ses.png")

# ── Save summary CSV ───────────────────────────────────────────────────────────
summary_df = pd.DataFrame([stats]).T.reset_index()
summary_df.columns = ["metric", "value"]
summary_df.to_csv(os.path.join(METRICS_DIR, "exploration_summary.csv"), index=False)
print(f"\nSaved exploration_summary.csv")
print("\n[01_data_exploration.py] Done.")
