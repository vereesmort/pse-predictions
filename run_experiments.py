"""
run_experiments.py
==================
Master experiment runner. Orchestrates the full pipeline:

  1. Load and filter data
  2. Build drug features (target profiles + PPI embeddings + fingerprints)
  3. Generate negatives (random and structured)
  4. Run all three splitting strategies
  5. Train and evaluate all models (LightGBM, XGBoost, CatBoost, RGCN)
  6. Ablation over feature modes and pair operators
  7. Save results and generate plots

Flags
-----
  --ses           : "all" | "rep"  (all 963 SEs or 12 representative ones)
  --models        : comma-separated subset, e.g. "lightgbm,xgboost"
  --feature-modes : comma-separated, e.g. "target_only,target+fp"
  --operators     : comma-separated pair operators
  --splits        : comma-separated splitting strategies
  --neg-strategy  : "random" | "structured"
  --skip-rgcn     : flag to skip RGCN (slow on CPU)
  --smiles-cache  : path to pre-fetched SMILES CSV (optional)

Example
-------
    # Quick run on 12 representative SEs, skip RGCN
    python run_experiments.py --ses rep --skip-rgcn

    # Full run (will take hours)
    python run_experiments.py --ses all
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from configs.config import (
    RAW_FILES, OUTPUT_DIR, METRICS_DIR, PLOTS_DIR, CACHE_DIR,
    MIN_PAIRS_PER_SE, REPRESENTATIVE_SES, RANDOM_SEED,
    SPLIT_STRATEGIES, DEFAULT_SPLIT, NEG_RATIO,
    PAIR_OPERATORS, DEFAULT_PAIR_OP,
)

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(CACHE_DIR,   exist_ok=True)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ses",           default="rep",
                   help="'all' or 'rep' (12 representative SEs)")
    p.add_argument("--models",        default="lightgbm,xgboost,catboost",
                   help="Comma-separated model names")
    p.add_argument("--feature-modes", default="target_only,fp_only,target+fp",
                   help="Comma-separated feature modes")
    p.add_argument("--operators",     default=DEFAULT_PAIR_OP,
                   help="Comma-separated pair operators for ablation")
    p.add_argument("--splits",        default=",".join(SPLIT_STRATEGIES),
                   help="Comma-separated splitting strategies")
    p.add_argument("--neg-strategy",  default="structured",
                   choices=["random", "structured"])
    p.add_argument("--skip-rgcn",     action="store_true")
    p.add_argument("--smiles-cache",  default=os.path.join(CACHE_DIR, "smiles.csv"))
    return p.parse_args()


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── 1. Load data ───────────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    combo   = pd.read_csv(RAW_FILES["combo"])
    targets = pd.read_csv(RAW_FILES["targets"])
    ppi     = pd.read_csv(RAW_FILES["ppi"])

    se_cts  = combo.groupby("Polypharmacy Side Effect").size()
    se_500  = se_cts[se_cts >= MIN_PAIRS_PER_SE].index
    combo_f = combo[combo["Polypharmacy Side Effect"].isin(se_500)].copy()

    all_drugs = sorted(pd.unique(combo_f[["STITCH 1", "STITCH 2"]].values.ravel()))
    all_ses   = sorted(se_500.tolist())

    se_ids = list(REPRESENTATIVE_SES.keys()) if args.ses == "rep" else all_ses
    print(f"  Drugs: {len(all_drugs)}, SEs: {len(se_ids)} "
          f"({'representative' if args.ses == 'rep' else 'all'})")

    # ── 2. Build drug features ─────────────────────────────────────────────────
    print("\n[2/7] Building drug features...")
    from features.drug_features import DrugFeatureBuilder

    builder = DrugFeatureBuilder(targets, ppi,
                                 morgan_radius=2, morgan_nbits=2048, ppi_hops=2)

    # Load or fetch SMILES
    smiles_dict = {}
    if os.path.exists(args.smiles_cache):
        sc = pd.read_csv(args.smiles_cache)
        smiles_dict = {r["STITCH"]: r["SMILES"]
                       for _, r in sc.iterrows() if pd.notna(r["SMILES"])}
        print(f"  Loaded {len(smiles_dict)} SMILES from cache")
    else:
        print("  No SMILES cache found — fingerprints will be zeros. "
              "Run features/drug_features.py::fetch_smiles_pubchem() to populate.")

    feature_set = builder.build_all(all_drugs, smiles_dict, ppi_embedding_dim=64)
    feature_set.coverage_report()

    # ── 3. Generate negatives ──────────────────────────────────────────────────
    print(f"\n[3/7] Generating {args.neg_strategy} negatives (ratio={NEG_RATIO})...")
    from preprocessing.sampling import NegativeSampler

    neg_cache = os.path.join(CACHE_DIR, f"negatives_{args.neg_strategy}.csv")
    if os.path.exists(neg_cache):
        negatives = pd.read_csv(neg_cache)
        print(f"  Loaded {len(negatives)} negatives from cache")
    else:
        sampler  = NegativeSampler(combo_f, strategy=args.neg_strategy, seed=RANDOM_SEED)
        negatives = sampler.sample_all(neg_ratio=NEG_RATIO, se_ids=se_ids)
        negatives.to_csv(neg_cache, index=False)
        print(f"  Generated and cached {len(negatives)} negatives")

    # ── 4. Build splits ────────────────────────────────────────────────────────
    print("\n[4/7] Building splits...")
    from preprocessing.splitting import Splitter

    split_strategies = [s.strip() for s in args.splits.split(",")]
    all_splits = {}   # {strategy: {se_id: SplitResult}}

    for strategy in split_strategies:
        print(f"  Strategy: {strategy}")
        splitter = Splitter(combo_f, negatives, strategy=strategy, seed=RANDOM_SEED)
        all_splits[strategy] = {se: splitter.split(se) for se in se_ids}

    # ── 5. Run models ──────────────────────────────────────────────────────────
    print("\n[5/7] Running models...")
    from evaluation.metrics import MetricsAggregator

    # Load per-SE metadata for aggregator
    meta_path = os.path.join(METRICS_DIR, "per_se_missing_targets.csv")
    if os.path.exists(meta_path):
        se_meta = pd.read_csv(meta_path)
    else:
        print("  Warning: per_se_missing_targets.csv not found. "
              "Run 01_data_exploration.py first.")
        se_meta = pd.DataFrame({"Polypharmacy Side Effect": se_ids,
                                 "Total Pairs": [0]*len(se_ids),
                                 "Pct Drugs Without Target": [0]*len(se_ids)})

    agg = MetricsAggregator(se_meta)

    model_names   = [m.strip() for m in args.models.split(",")]
    feature_modes = [f.strip() for f in args.feature_modes.split(",")]
    operators     = [o.strip() for o in args.operators.split(",")]
    use_strategy  = DEFAULT_SPLIT if DEFAULT_SPLIT in split_strategies else split_strategies[0]

    all_result_frames = []

    for strategy in split_strategies:
        splits = all_splits[strategy]

        for feature_mode in feature_modes:
            for operator in operators:

                if "lightgbm" in model_names:
                    print(f"  LightGBM | {feature_mode} | {operator} | {strategy}")
                    from models.lightgbm_model import LGBMPredictor
                    lgbm = LGBMPredictor(feature_set, operator=operator,
                                         feature_mode=feature_mode)
                    res = lgbm.run(splits, se_ids=se_ids)
                    all_result_frames.append(res)
                    agg.add(res)

                if "xgboost" in model_names:
                    print(f"  XGBoost  | {feature_mode} | {operator} | {strategy}")
                    from models.xgboost_model import XGBPredictor
                    xgb = XGBPredictor(feature_set, operator=operator,
                                       feature_mode=feature_mode)
                    res = xgb.run(splits, se_ids=se_ids)
                    all_result_frames.append(res)
                    agg.add(res)

                if "catboost" in model_names:
                    print(f"  CatBoost | {feature_mode} | {operator} | {strategy}")
                    from models.catboost_model import CatBoostPredictor
                    cb = CatBoostPredictor(feature_set, operator=operator,
                                          feature_mode=feature_mode)
                    res = cb.run(splits, se_ids=se_ids)
                    all_result_frames.append(res)
                    agg.add(res)

    # RGCN — trained once, evaluated per feature mode
    if not args.skip_rgcn and "rgcn" in model_names:
        print("\n  RGCN (relational GNN)...")
        from models.rgcn_model import RGCNPredictor

        for feature_mode in feature_modes:
            print(f"  RGCN | {feature_mode}")
            # Build combined train/val for RGCN graph construction
            strategy    = use_strategy
            splits_rgcn = all_splits[strategy]
            train_frames = [splits_rgcn[se].train for se in se_ids
                            if len(splits_rgcn[se].train) > 0]
            val_frames   = [splits_rgcn[se].val   for se in se_ids
                            if len(splits_rgcn[se].val)   > 0]
            train_all = pd.concat(train_frames, ignore_index=True)
            val_all   = pd.concat(val_frames,   ignore_index=True)

            rgcn = RGCNPredictor(feature_set, all_drugs, all_ses)
            rgcn.train(combo_f, train_all, val_all, feature_mode=feature_mode)

            res = rgcn.evaluate_all(splits_rgcn, combo_f,
                                    se_ids=se_ids, feature_mode=feature_mode)
            all_result_frames.append(res)
            agg.add(res)

    # ── 6. Save all results ────────────────────────────────────────────────────
    print("\n[6/7] Saving results...")
    if all_result_frames:
        all_res = pd.concat(all_result_frames, ignore_index=True)
        all_res.to_csv(os.path.join(METRICS_DIR, "all_results.csv"), index=False)
        print(f"  Saved all_results.csv ({len(all_res)} rows)")

    # ── 7. Generate plots ──────────────────────────────────────────────────────
    print("\n[7/7] Generating plots...")
    from evaluation.reporting import Reporter
    reporter = Reporter(agg)
    reporter.plot_all()
    reporter.save_tables()

    print("\n=== Experiment complete ===")
    print(f"Results in: {METRICS_DIR}")
    print(f"Plots in:   {PLOTS_DIR}")


if __name__ == "__main__":
    main()
