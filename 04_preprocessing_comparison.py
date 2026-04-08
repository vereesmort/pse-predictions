"""
04_preprocessing_comparison.py
================================
Runs a controlled comparison of all preprocessing choices on the 12
representative side effects and produces a summary report:

  - Splitting strategy: random_pair vs stratified_pair vs drug_cold_start
  - Negative sampling: random vs structured
  - Pair operator: hadamard, absdiff, hadamard_absdiff, sum, concat
  - Feature mode: target_only, fp_only, target+fp

Model fixed to LightGBM for speed.
All combinations → results/metrics/preprocessing_comparison.csv

Run:
    python 04_preprocessing_comparison.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

from configs.config import (
    RAW_FILES, METRICS_DIR, PLOTS_DIR, CACHE_DIR,
    MIN_PAIRS_PER_SE, REPRESENTATIVE_SES, RANDOM_SEED,
    NEG_RATIO, PAIR_OPERATORS,
)
from features.drug_features import DrugFeatureBuilder
from preprocessing.sampling import NegativeSampler
from preprocessing.splitting import Splitter
from models.lightgbm_model import LGBMPredictor
from evaluation.metrics import compute_metrics

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def run_preprocessing_comparison():
    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data...")
    combo   = pd.read_csv(RAW_FILES["combo"])
    targets = pd.read_csv(RAW_FILES["targets"])
    ppi     = pd.read_csv(RAW_FILES["ppi"])

    se_cts  = combo.groupby("Polypharmacy Side Effect").size()
    se_500  = se_cts[se_cts >= MIN_PAIRS_PER_SE].index
    combo_f = combo[combo["Polypharmacy Side Effect"].isin(se_500)].copy()

    all_drugs = sorted(pd.unique(combo_f[["STITCH 1","STITCH 2"]].values.ravel()))
    se_ids    = list(REPRESENTATIVE_SES.keys())

    # ── Build features ─────────────────────────────────────────────────────────
    print("Building drug features...")
    builder = DrugFeatureBuilder(targets, ppi)
    smiles_cache = os.path.join(CACHE_DIR, "smiles.csv")
    smiles_dict  = {}
    if os.path.exists(smiles_cache):
        sc = pd.read_csv(smiles_cache)
        smiles_dict = {r["STITCH"]: r["SMILES"] for _, r in sc.iterrows()
                       if pd.notna(r["SMILES"])}
    feature_set = builder.build_all(all_drugs, smiles_dict, ppi_embedding_dim=64)

    # ── Experiment grid ────────────────────────────────────────────────────────
    split_strategies  = ["random_pair", "stratified_pair", "drug_cold_start"]
    neg_strategies    = ["random", "structured"]
    feature_modes     = ["target_only", "fp_only", "target+fp"]
    operators         = ["hadamard", "absdiff", "hadamard_absdiff", "sum", "concat"]

    rows = []
    total = len(split_strategies) * len(neg_strategies) * len(feature_modes) * len(operators)
    run   = 0

    for neg_strat in neg_strategies:
        # Generate or load negatives for this sampling strategy
        neg_cache = os.path.join(CACHE_DIR, f"negatives_{neg_strat}_rep.csv")
        if os.path.exists(neg_cache):
            negatives = pd.read_csv(neg_cache)
        else:
            print(f"  Sampling {neg_strat} negatives...")
            sampler   = NegativeSampler(combo_f, strategy=neg_strat, seed=RANDOM_SEED)
            negatives = sampler.sample_all(neg_ratio=NEG_RATIO, se_ids=se_ids)
            negatives.to_csv(neg_cache, index=False)

        for split_strat in split_strategies:
            splitter = Splitter(combo_f, negatives,
                                strategy=split_strat, seed=RANDOM_SEED)
            splits   = {se: splitter.split(se) for se in se_ids}

            for feat_mode in feature_modes:
                for operator in operators:
                    run += 1
                    print(f"  [{run}/{total}] neg={neg_strat} split={split_strat} "
                          f"feat={feat_mode} op={operator}")

                    model = LGBMPredictor(feature_set, operator=operator,
                                          feature_mode=feat_mode)
                    res   = model.run(splits, se_ids=se_ids, verbose_every=999)

                    for _, row in res.iterrows():
                        rows.append({
                            "neg_strategy"  : neg_strat,
                            "split_strategy": split_strat,
                            "feature_mode"  : feat_mode,
                            "operator"      : operator,
                            **{k: v for k, v in row.items()},
                        })

    # ── Save ───────────────────────────────────────────────────────────────────
    results = pd.DataFrame(rows)
    out     = os.path.join(METRICS_DIR, "preprocessing_comparison.csv")
    results.to_csv(out, index=False)
    print(f"\nSaved {len(results)} rows → {out}")

    # ── Summary tables ─────────────────────────────────────────────────────────
    group_cols = ["neg_strategy", "split_strategy", "feature_mode", "operator"]
    summary = (results.groupby(group_cols)[["auroc","ap"]]
                      .mean().round(4).reset_index()
                      .sort_values("auroc", ascending=False))
    print("\n=== Top 10 configurations by AUROC ===")
    print(summary.head(10).to_string(index=False))

    # ── Plot: splitting strategy inflation ─────────────────────────────────────
    _plot_split_inflation(results)

    # ── Plot: operator comparison ──────────────────────────────────────────────
    _plot_operator_comparison(results)

    # ── Plot: feature mode ablation ────────────────────────────────────────────
    _plot_feature_mode(results)

    return results


def _plot_split_inflation(results):
    """Show AUROC distribution per splitting strategy."""
    fig, ax = plt.subplots(figsize=(9, 5))
    strategies = ["random_pair", "stratified_pair", "drug_cold_start"]
    colors     = ["#E24B4A", "#EF9F27", "#1D9E75"]
    labels     = ["Random pair", "Stratified pair", "Drug cold-start"]

    # Use target+fp, hadamard_absdiff, structured negatives as the fixed condition
    sub = results[
        (results["feature_mode"] == "target+fp") &
        (results["operator"] == "hadamard_absdiff") &
        (results["neg_strategy"] == "structured")
    ]

    data = [sub[sub["split_strategy"] == s]["auroc"].dropna().values
            for s in strategies]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=1.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(labels)
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC distribution by splitting strategy\n"
                 "(LightGBM, target+fp, hadamard_absdiff, structured negatives)")
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "preproc_split_inflation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def _plot_operator_comparison(results):
    """Bar chart of mean AUROC per pair operator."""
    sub = results[
        (results["split_strategy"] == "drug_cold_start") &
        (results["feature_mode"] == "target+fp") &
        (results["neg_strategy"] == "structured")
    ]
    summary = (sub.groupby("operator")[["auroc","ap"]]
                  .mean().round(4).reset_index()
                  .sort_values("auroc", ascending=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#378ADD","#1D9E75","#D85A30","#7F77DD","#D4537E"]
    for ax, metric in zip(axes, ["auroc","ap"]):
        bars = ax.bar(summary["operator"], summary[metric],
                      color=colors[:len(summary)], alpha=0.85)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Pair operator comparison — {metric.upper()}")
        ax.set_ylim(max(0, summary[metric].min() - 0.05),
                    min(1, summary[metric].max() + 0.05))
        for bar, val in zip(bars, summary[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "preproc_operator_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def _plot_feature_mode(results):
    """Bar chart: feature mode ablation (target_only vs fp_only vs target+fp)."""
    sub = results[
        (results["split_strategy"] == "drug_cold_start") &
        (results["operator"] == "hadamard_absdiff") &
        (results["neg_strategy"] == "structured")
    ]
    summary = (sub.groupby("feature_mode")[["auroc","ap"]]
                  .mean().round(4).reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors_map = {"target_only": "#378ADD", "fp_only": "#D85A30", "target+fp": "#1D9E75"}
    for ax, metric in zip(axes, ["auroc","ap"]):
        bars = ax.bar(summary["feature_mode"],
                      summary[metric],
                      color=[colors_map.get(m, "#888") for m in summary["feature_mode"]],
                      alpha=0.85)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Feature mode ablation — {metric.upper()}")
        ax.set_ylim(max(0, summary[metric].min() - 0.05),
                    min(1, summary[metric].max() + 0.05))
        for bar, val in zip(bars, summary[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "preproc_feature_mode.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    run_preprocessing_comparison()
