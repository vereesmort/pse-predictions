"""
models/reproductions/run_all_comparisons.py
============================================
Master script: run all four paper reproductions under both original
(inflated) and fair protocols, then produce a unified comparison table.

Produces:
  results/metrics/repro_all_comparison.csv
  results/metrics/repro_inflation_summary.csv

Each paper is run with default settings. Pass --ses all for full 963 SEs
(hours); default is the 12 representative SEs.

Usage
-----
    python models/reproductions/run_all_comparisons.py
    python models/reproductions/run_all_comparisons.py --ses all
    python models/reproductions/run_all_comparisons.py --skip nnps_orig
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np

from models.reproductions._utils import (
    save_results, compare_protocols, SE_NAMES,
)
from configs.config import METRICS_DIR


def run_all(se_ids_arg="rep", skip=None):
    skip  = set(skip or [])
    all_results = []

    print("\n" + "█"*65)
    print("RUNNING ALL PAPER REPRODUCTIONS")
    print(f"SE set: {se_ids_arg}   Skip: {skip or 'none'}")
    print("█"*65)

    # ── NNPS ──────────────────────────────────────────────────────────────────
    if "nnps_orig" not in skip:
        print("\n\n>>> NNPS — ORIGINAL (inflated)")
        from models.reproductions.nnps_original import run_nnps_original
        r = run_nnps_original(se_ids=None if se_ids_arg=="rep" else "all")
        r["paper"] = "NNPS"; r["protocol_type"] = "original"
        all_results.append(r)

    if "nnps_fair" not in skip:
        print("\n\n>>> NNPS — FAIR")
        from models.reproductions.nnps_fair import run_nnps_fair
        r = run_nnps_fair(se_ids=None if se_ids_arg=="rep" else "all")
        r["paper"] = "NNPS"; r["protocol_type"] = "fair"
        all_results.append(r)

    # ── Decagon / RGCN ────────────────────────────────────────────────────────
    if "decagon_orig" not in skip:
        print("\n\n>>> DECAGON (RGCN) — ORIGINAL (random split)")
        from models.reproductions.decagon_rgcn import run_decagon_rgcn
        r = run_decagon_rgcn(
            se_ids=None if se_ids_arg=="rep" else "all",
            split_strategy="random_pair", neg_strategy="random",
            feature_mode="target+fp")
        r["paper"] = "Decagon"; r["protocol_type"] = "original"
        all_results.append(r)

    if "decagon_fair" not in skip:
        print("\n\n>>> DECAGON (RGCN) — FAIR (cold-start)")
        from models.reproductions.decagon_rgcn import run_decagon_rgcn
        r = run_decagon_rgcn(
            se_ids=None if se_ids_arg=="rep" else "all",
            split_strategy="drug_cold_start", neg_strategy="structured",
            feature_mode="target+fp")
        r["paper"] = "Decagon"; r["protocol_type"] = "fair"
        all_results.append(r)

    # ── TF-Decagon ────────────────────────────────────────────────────────────
    if "tf_orig" not in skip:
        print("\n\n>>> TF-DECAGON — ORIGINAL (transductive)")
        from models.reproductions.tf_decagon_transductive import \
            run_tf_decagon_transductive
        r = run_tf_decagon_transductive(
            se_ids=None if se_ids_arg=="rep" else "all",
            model_name="distmult", dim=128, epochs=30, add_self_loops=True)
        r["paper"] = "TF-Decagon"; r["protocol_type"] = "original"
        all_results.append(r)

    if "tf_fair" not in skip:
        print("\n\n>>> TF-DECAGON — FAIR (inductive cold-start)")
        from models.reproductions.tf_decagon_inductive import \
            run_tf_decagon_inductive
        r = run_tf_decagon_inductive(
            se_ids=None if se_ids_arg=="rep" else "all",
            model_name="distmult", dim=128, epochs=30,
            add_self_loops=True, feature_mode="target+fp")
        r["paper"] = "TF-Decagon"; r["protocol_type"] = "fair"
        all_results.append(r)

    # ── SimVec ────────────────────────────────────────────────────────────────
    if "simvec_orig" not in skip:
        print("\n\n>>> SIMVEC — ORIGINAL (weak-node split)")
        from models.reproductions.simvec_weak_node import run_simvec
        r = run_simvec(
            se_ids=None if se_ids_arg=="rep" else "all",
            split_type="weak_node", dim=100, epochs=20)
        r["paper"] = "SimVec"; r["protocol_type"] = "original"
        all_results.append(r)

    if "simvec_fair" not in skip:
        print("\n\n>>> SIMVEC — FAIR (drug cold-start)")
        from models.reproductions.simvec_weak_node import run_simvec
        r = run_simvec(
            se_ids=None if se_ids_arg=="rep" else "all",
            split_type="drug_cold_start", dim=100, epochs=20)
        r["paper"] = "SimVec"; r["protocol_type"] = "fair"
        all_results.append(r)

    if not all_results:
        print("No results collected — check --skip flags.")
        return

    # ── Combine and summarise ─────────────────────────────────────────────────
    combined = pd.concat(all_results, ignore_index=True)
    combined["se_name"] = combined["se_id"].map(SE_NAMES).fillna(combined["se_id"])
    save_results(combined, "all_comparison")

    # Summary table: mean AUROC per paper × protocol
    print("\n\n" + "="*65)
    print("FULL COMPARISON SUMMARY — Mean AUROC across SEs")
    print("="*65)
    summary = (combined.groupby(["paper", "protocol_type"])["auroc"]
                       .mean().round(4).unstack())
    if "original" in summary.columns and "fair" in summary.columns:
        summary["inflation"] = (summary["original"] - summary["fair"]).round(4)
    print(summary.to_string())

    # Inflation table
    print("\n" + "="*65)
    print("AUROC INFLATION (original − fair) — the measurement artefact")
    print("="*65)
    if "inflation" in summary.columns:
        for paper, row in summary.iterrows():
            if pd.notna(row.get("inflation")):
                print(f"  {paper:15s}: original={row.get('original','N/A'):.4f}  "
                      f"fair={row.get('fair','N/A'):.4f}  "
                      f"inflation=+{row['inflation']:.4f}")

    # Save inflation summary
    inflation_path = os.path.join(METRICS_DIR, "repro_inflation_summary.csv")
    summary.reset_index().to_csv(inflation_path, index=False)
    print(f"\nSaved inflation summary → {inflation_path}")
    print(f"Saved all results      → {METRICS_DIR}/repro_all_comparison.csv")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ses",  default="rep",
                        help="'rep' (12 SEs) or 'all' (963 SEs)")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Papers to skip: nnps_orig nnps_fair decagon_orig "
                             "decagon_fair tf_orig tf_fair simvec_orig simvec_fair")
    args = parser.parse_args()
    run_all(se_ids_arg=args.ses, skip=args.skip)
