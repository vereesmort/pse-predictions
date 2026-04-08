"""
05_ablation.py
==============
Structured ablation study for the research paper.

Ablation dimensions
-------------------
A) Feature mode ablation per target-coverage regime
   For each drug pair, classify as:
     - "both_have"   : both drugs have protein targets
     - "one_has"     : exactly one drug has a protein target
     - "neither_has" : neither drug has a protein target

   Compare AUROC/AP across modes: target_only, fp_only, target+fp
   Report the ΔAP from adding fingerprints (fingerprint contribution).

B) Pair operator ablation
   Fix: drug_cold_start split, structured negatives, target+fp
   Vary: operator across all 5 options

C) Splitting strategy comparison (per SE)
   For each representative SE, show AUROC under random vs cold-start split
   with the same model/features — quantifies the inflation.

D) Negative sampling comparison
   Fix everything else; compare random vs structured negatives.

Output
------
  results/metrics/ablation_*.csv
  results/plots/ablation_*.png

Run:
    python 05_ablation.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configs.config import (
    RAW_FILES, METRICS_DIR, PLOTS_DIR, CACHE_DIR,
    MIN_PAIRS_PER_SE, REPRESENTATIVE_SES, RANDOM_SEED, NEG_RATIO,
)
from features.drug_features import DrugFeatureBuilder
from preprocessing.sampling import NegativeSampler
from preprocessing.splitting import Splitter
from models.lightgbm_model import LGBMPredictor
from evaluation.metrics import compute_metrics

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_data():
    combo   = pd.read_csv(RAW_FILES["combo"])
    targets = pd.read_csv(RAW_FILES["targets"])
    ppi     = pd.read_csv(RAW_FILES["ppi"])
    se_cts  = combo.groupby("Polypharmacy Side Effect").size()
    se_500  = se_cts[se_cts >= MIN_PAIRS_PER_SE].index
    combo_f = combo[combo["Polypharmacy Side Effect"].isin(se_500)].copy()
    return combo_f, targets, ppi


def assign_target_regime(df: pd.DataFrame, t_drugs: set) -> pd.Series:
    """
    For each row in df (with drug_1, drug_2 columns), assign:
      "both_have"    if both drugs are in t_drugs
      "one_has"      if exactly one is
      "neither_has"  if neither is
    """
    d1_has = df["drug_1"].isin(t_drugs)
    d2_has = df["drug_2"].isin(t_drugs)
    regime = np.where(d1_has & d2_has, "both_have",
             np.where(d1_has | d2_has, "one_has", "neither_has"))
    return pd.Series(regime, index=df.index)


# ── Ablation A: Feature mode × target regime ───────────────────────────────────

def ablation_feature_mode_by_regime(combo_f, feature_set, t_drugs, splits, se_ids):
    print("\n=== Ablation A: Feature mode × target regime ===")
    feature_modes = ["target_only", "fp_only", "target+fp"]
    rows = []

    for feat_mode in feature_modes:
        print(f"  Feature mode: {feat_mode}")
        model = LGBMPredictor(feature_set, operator="hadamard_absdiff",
                              feature_mode=feat_mode)
        for se_id in se_ids:
            split = splits[se_id]
            if len(split.train) == 0 or split.train["label"].nunique() < 2:
                continue
            model.fit_one(se_id, split.train, split.val)

            # Evaluate per regime on test set
            test = split.test.copy()
            test["regime"] = assign_target_regime(test, t_drugs)

            for regime in ["both_have", "one_has", "neither_has"]:
                sub = test[test["regime"] == regime]
                if len(sub) < 10 or sub["label"].nunique() < 2:
                    continue
                proba   = model.predict_proba(se_id, sub)
                metrics = compute_metrics(sub["label"].values, proba)
                rows.append({
                    "se_id"       : se_id,
                    "se_name"     : REPRESENTATIVE_SES.get(se_id, se_id),
                    "feature_mode": feat_mode,
                    "regime"      : regime,
                    "n_pairs"     : len(sub),
                    **metrics,
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(METRICS_DIR, "ablation_A_feature_regime.csv"), index=False)
    print(f"  Saved ablation_A_feature_regime.csv ({len(df)} rows)")

    # Summary: mean AUROC per (feature_mode, regime)
    summary = df.groupby(["feature_mode","regime"])[["auroc","ap"]].mean().round(4)
    print(summary)

    # ΔAP: improvement from target+fp over target_only per regime
    pivot = df.pivot_table(values="ap", index=["se_id","regime"],
                           columns="feature_mode", aggfunc="mean")
    if "target+fp" in pivot.columns and "target_only" in pivot.columns:
        pivot["delta_ap_fp"] = pivot["target+fp"] - pivot["target_only"]
        delta_summary = pivot.groupby("regime")["delta_ap_fp"].mean().round(4)
        print("\n  ΔAP (target+fp − target_only) by regime:")
        print(delta_summary)
        delta_summary.to_csv(os.path.join(METRICS_DIR, "ablation_A_delta_ap.csv"))

    _plot_ablation_A(df)
    return df


def _plot_ablation_A(df):
    regimes   = ["both_have", "one_has", "neither_has"]
    modes     = ["target_only", "fp_only", "target+fp"]
    colors    = {"target_only": "#378ADD", "fp_only": "#D85A30", "target+fp": "#1D9E75"}
    x         = np.arange(len(regimes))
    w         = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, ["auroc", "ap"]):
        for i, mode in enumerate(modes):
            sub  = df[df["feature_mode"] == mode]
            vals = [sub[sub["regime"] == r][metric].mean() for r in regimes]
            ax.bar(x + i*w, vals, w, label=mode,
                   color=colors[mode], alpha=0.85)
        ax.set_xticks(x + w)
        ax.set_xticklabels(["Both\nhave target", "One has\ntarget",
                             "Neither has\ntarget"])
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Feature mode × target regime — {metric.upper()}")
        ax.legend(fontsize=9)
        ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "ablation_A_feature_regime.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ── Ablation B: Pair operator ──────────────────────────────────────────────────

def ablation_pair_operator(feature_set, splits, se_ids):
    print("\n=== Ablation B: Pair operator ===")
    operators = ["hadamard", "absdiff", "hadamard_absdiff", "sum", "concat"]
    rows = []

    for op in operators:
        print(f"  Operator: {op}")
        model = LGBMPredictor(feature_set, operator=op, feature_mode="target+fp")
        res   = model.run(splits, se_ids=se_ids, verbose_every=999)
        res["operator"] = op
        rows.append(res)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(METRICS_DIR, "ablation_B_operator.csv"), index=False)

    summary = df.groupby("operator")[["auroc","ap"]].mean().round(4).sort_values("auroc", ascending=False)
    print(summary)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ["#378ADD","#1D9E75","#D85A30","#7F77DD","#D4537E"]
    for ax, metric in zip(axes, ["auroc","ap"]):
        vals = [summary.loc[op, metric] if op in summary.index else 0 for op in operators]
        bars = ax.bar(operators, vals, color=colors, alpha=0.85)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Pair operator ablation — {metric.upper()}")
        ax.set_ylim(max(0, min(vals) - 0.04), min(1, max(vals) + 0.04))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "ablation_B_operator.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")
    return df


# ── Ablation C: Splitting strategy inflation per SE ────────────────────────────

def ablation_split_inflation(combo_f, feature_set, all_neg, se_ids):
    print("\n=== Ablation C: Splitting strategy inflation ===")
    strategies = ["random_pair", "stratified_pair", "drug_cold_start"]
    rows = []

    for strategy in strategies:
        splitter = Splitter(combo_f, all_neg, strategy=strategy, seed=RANDOM_SEED)
        splits   = {se: splitter.split(se) for se in se_ids}
        model    = LGBMPredictor(feature_set, operator="hadamard_absdiff",
                                 feature_mode="target+fp")
        res = model.run(splits, se_ids=se_ids, verbose_every=999)
        res["split_strategy"] = strategy
        rows.append(res)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(METRICS_DIR, "ablation_C_split_inflation.csv"), index=False)

    # Per-SE AUROC comparison
    pivot = df.pivot_table(values="auroc", index="se_id",
                           columns="split_strategy", aggfunc="mean").round(4)
    if "random_pair" in pivot.columns and "drug_cold_start" in pivot.columns:
        pivot["inflation"] = pivot["random_pair"] - pivot["drug_cold_start"]
        print("\n  Inflation (random_pair AUROC − drug_cold_start AUROC) per SE:")
        print(pivot[["random_pair","drug_cold_start","inflation"]].to_string())

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_map = {"random_pair": "#E24B4A", "stratified_pair": "#EF9F27",
                  "drug_cold_start": "#1D9E75"}
    ses_sorted = sorted(se_ids)
    for strat in strategies:
        sub  = df[df["split_strategy"] == strat].set_index("se_id")
        vals = [sub.loc[se, "auroc"] if se in sub.index else np.nan for se in ses_sorted]
        ax.plot(range(len(ses_sorted)), vals, marker="o", markersize=5,
                label=strat, color=colors_map[strat], alpha=0.8)
    ax.set_xticks(range(len(ses_sorted)))
    ax.set_xticklabels([REPRESENTATIVE_SES.get(s, s)[:15] for s in ses_sorted],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("AUROC")
    ax.set_title("Splitting strategy inflation across 12 representative SEs")
    ax.legend(fontsize=9)
    ax.set_ylim(0.4, 1.0)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "ablation_C_split_inflation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")
    return df


# ── Ablation D: Negative sampling strategy ────────────────────────────────────

def ablation_neg_sampling(combo_f, feature_set, splits_by_neg, se_ids):
    print("\n=== Ablation D: Negative sampling strategy ===")
    rows = []
    for neg_strat, splits in splits_by_neg.items():
        model = LGBMPredictor(feature_set, operator="hadamard_absdiff",
                              feature_mode="target+fp")
        res = model.run(splits, se_ids=se_ids, verbose_every=999)
        res["neg_strategy"] = neg_strat
        rows.append(res)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(os.path.join(METRICS_DIR, "ablation_D_neg_sampling.csv"), index=False)

    summary = df.groupby("neg_strategy")[["auroc","ap","f1"]].mean().round(4)
    print(summary)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    strategies = ["random", "structured"]
    colors = ["#378ADD", "#D85A30"]
    for ax, metric in zip(axes, ["auroc","ap"]):
        vals = [summary.loc[s, metric] if s in summary.index else 0 for s in strategies]
        bars = ax.bar(strategies, vals, color=colors, alpha=0.85, width=0.4)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Negative sampling — {metric.upper()}")
        ax.set_ylim(max(0, min(vals) - 0.05), min(1, max(vals) + 0.05))
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "ablation_D_neg_sampling.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    combo_f, targets, ppi = load_data()
    all_drugs = sorted(pd.unique(combo_f[["STITCH 1","STITCH 2"]].values.ravel()))
    se_ids    = list(REPRESENTATIVE_SES.keys())
    t_drugs   = set(targets["STITCH"])

    print("Building features...")
    builder = DrugFeatureBuilder(targets, ppi)
    smiles_cache = os.path.join(CACHE_DIR, "smiles.csv")
    smiles_dict  = {}
    if os.path.exists(smiles_cache):
        sc = pd.read_csv(smiles_cache)
        smiles_dict = {r["STITCH"]: r["SMILES"] for _, r in sc.iterrows()
                       if pd.notna(r["SMILES"])}
    feature_set = builder.build_all(all_drugs, smiles_dict, ppi_embedding_dim=64)

    # Generate both negative sets
    splits_by_neg = {}
    for neg_strat in ["random", "structured"]:
        neg_cache = os.path.join(CACHE_DIR, f"negatives_{neg_strat}_rep.csv")
        if os.path.exists(neg_cache):
            negatives = pd.read_csv(neg_cache)
        else:
            sampler   = NegativeSampler(combo_f, strategy=neg_strat, seed=RANDOM_SEED)
            negatives = sampler.sample_all(neg_ratio=NEG_RATIO, se_ids=se_ids)
            negatives.to_csv(neg_cache, index=False)

        splitter = Splitter(combo_f, negatives, strategy="drug_cold_start",
                            seed=RANDOM_SEED)
        splits_by_neg[neg_strat] = {se: splitter.split(se) for se in se_ids}

    # Use structured negatives + cold-start as primary for ablations A & B
    primary_splits = splits_by_neg["structured"]
    struct_neg_all = pd.read_csv(
        os.path.join(CACHE_DIR, "negatives_structured_rep.csv")
    )

    ablation_feature_mode_by_regime(combo_f, feature_set, t_drugs,
                                    primary_splits, se_ids)
    ablation_pair_operator(feature_set, primary_splits, se_ids)
    ablation_split_inflation(combo_f, feature_set, struct_neg_all, se_ids)
    ablation_neg_sampling(combo_f, feature_set, splits_by_neg, se_ids)

    print("\n[05_ablation.py] All ablations complete.")
