"""
models/reproductions/_utils.py
==============================
Shared utilities for all reproduction scripts.

Provides:
  - load_decagon_data()       — load and filter the 5 raw CSVs
  - build_feature_set()       — construct DrugFeatureSet for a drug list
  - build_negatives()         — generate negatives with given strategy
  - build_splits()            — generate splits with given strategy
  - run_per_se_gbt()          — generic per-SE GBT evaluation loop
  - save_results()            — save results CSV and print summary
  - compare_protocols()       — print side-by-side original vs fair results
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd

from configs.config import (
    RAW_FILES, MIN_PAIRS_PER_SE, METRICS_DIR, CACHE_DIR,
    RANDOM_SEED, NEG_RATIO, REPRESENTATIVE_SES,
)
from preprocessing.sampling import NegativeSampler
from preprocessing.splitting import Splitter
from evaluation.metrics import compute_metrics

SE_NAMES = {
    "C0034065": "Pulmonary embolism",
    "C0020473": "Hyperlipaemia",
    "C1510472": "Drug addiction",
    "C0001824": "Agranulocytosis",
    "C0019348": "Herpes simplex",
    "C0085606": "Micturition urgency",
    "C0025289": "Meningitis",
    "C1510431": "Superficial thrombophlebitis",
    "C0029878": "External ear infection",
    "C0243010": "Viral encephalitis",
    "C0042164": "Intraocular inflammation",
    "C0332687": "Burns second degree",
}


# ── Data loading ───────────────────────────────────────────────────────────────

def load_decagon_data(se_ids=None):
    """
    Load and filter the Decagon raw CSVs.

    Returns
    -------
    combo_f   : filtered combo DataFrame (≥500 pairs per SE)
    mono      : mono side effects DataFrame
    ppi       : PPI DataFrame
    targets   : drug–protein targets DataFrame
    all_drugs : sorted list of unique drug STITCH IDs in combo_f
    se_ids    : list of SE CUI strings to use (defaults to 12 representative SEs)
    """
    print("Loading Decagon data...")
    combo   = pd.read_csv(RAW_FILES["combo"])
    mono    = pd.read_csv(RAW_FILES["mono"])
    ppi     = pd.read_csv(RAW_FILES["ppi"])
    targets = pd.read_csv(RAW_FILES["targets"])

    se_cts  = combo.groupby("Polypharmacy Side Effect").size()
    se_500  = se_cts[se_cts >= MIN_PAIRS_PER_SE].index
    combo_f = combo[combo["Polypharmacy Side Effect"].isin(se_500)].copy()

    all_drugs = sorted(pd.unique(combo_f[["STITCH 1", "STITCH 2"]].values.ravel()))

    if se_ids is None:
        se_ids = list(REPRESENTATIVE_SES.keys())

    print(f"  Drugs: {len(all_drugs)}, SEs: {len(se_ids)}, "
          f"combo pairs: {len(combo_f):,}")
    return combo_f, mono, ppi, targets, all_drugs, se_ids


# ── Feature building ───────────────────────────────────────────────────────────

def build_feature_set(targets, ppi, all_drugs, smiles_dict=None,
                      ppi_dim=64, fp_dim=2048):
    """Build DrugFeatureSet. Loads SMILES from cache if smiles_dict not given."""
    from features.drug_features import DrugFeatureBuilder

    if smiles_dict is None:
        smiles_cache = os.path.join(CACHE_DIR, "smiles.csv")
        if os.path.exists(smiles_cache):
            sc = pd.read_csv(smiles_cache)
            smiles_dict = {r["STITCH"]: r["SMILES"]
                           for _, r in sc.iterrows() if pd.notna(r["SMILES"])}
            print(f"  Loaded {len(smiles_dict)} SMILES from cache.")
        else:
            smiles_dict = {}
            print("  No SMILES cache — fingerprints will be zero.")

    builder = DrugFeatureBuilder(targets, ppi, morgan_nbits=fp_dim)
    fset    = builder.build_all(all_drugs, smiles_dict, ppi_embedding_dim=ppi_dim)
    fset.coverage_report()
    return fset


# ── Negatives & splits ─────────────────────────────────────────────────────────

def build_negatives(combo_f, strategy, se_ids, neg_ratio=NEG_RATIO,
                    seed=RANDOM_SEED):
    """Generate or load cached negatives."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR,
                              f"negatives_{strategy}_{'rep' if len(se_ids)==12 else 'all'}.csv")
    if os.path.exists(cache_path):
        neg = pd.read_csv(cache_path)
        print(f"  Loaded {len(neg):,} {strategy} negatives from cache.")
        return neg

    print(f"  Sampling {strategy} negatives (ratio={neg_ratio})...")
    sampler = NegativeSampler(combo_f, strategy=strategy, seed=seed)
    neg     = sampler.sample_all(neg_ratio=neg_ratio, se_ids=se_ids)
    neg.to_csv(cache_path, index=False)
    print(f"  Saved {len(neg):,} negatives → {cache_path}")
    return neg


def build_splits(combo_f, negatives, strategy, se_ids, seed=RANDOM_SEED):
    """Build split dict {se_id: SplitResult}."""
    splitter = Splitter(combo_f, negatives, strategy=strategy, seed=seed)
    return {se: splitter.split(se) for se in se_ids}


# ── Generic GBT evaluation loop ───────────────────────────────────────────────

def run_per_se_gbt(model_cls, model_kwargs, feature_set, splits,
                   se_ids, pair_operator, feature_mode, protocol_label,
                   verbose_every=3):
    """
    Train a per-SE GBT model and return results DataFrame.

    model_cls    : LGBMPredictor | XGBPredictor | CatBoostPredictor
    model_kwargs : dict of keyword arguments for model_cls
    """
    model = model_cls(feature_set, operator=pair_operator,
                      feature_mode=feature_mode, **model_kwargs)
    results = model.run(splits, se_ids=se_ids, verbose_every=verbose_every)
    results["protocol"] = protocol_label
    return results


# ── Save & report ──────────────────────────────────────────────────────────────

def save_results(results: pd.DataFrame, paper_name: str,
                 output_dir: str = None) -> str:
    """Save results CSV and print a summary table."""
    os.makedirs(METRICS_DIR, exist_ok=True)
    out_dir = output_dir or METRICS_DIR
    os.makedirs(out_dir, exist_ok=True)

    fname = os.path.join(out_dir,
                         f"repro_{paper_name.lower().replace(' ', '_')}.csv")
    results.to_csv(fname, index=False)

    # Add SE names
    df = results.copy()
    df["se_name"] = df["se_id"].map(SE_NAMES).fillna(df["se_id"])

    print(f"\n=== {paper_name} — Results ===")
    summary = (df.groupby(["protocol", "model"] if "model" in df.columns
                           else ["protocol"])
                 [["auroc", "ap", "f1"]]
                 .mean().round(4))
    print(summary.to_string())
    print(f"\nSaved → {fname}")
    return fname


def compare_protocols(results: pd.DataFrame, paper_name: str):
    """
    Print side-by-side comparison of original (inflated) vs fair protocol,
    and compute the inflation gap.
    """
    df = results.copy()
    df["se_name"] = df["se_id"].map(SE_NAMES).fillna(df["se_id"])

    protocols = df["protocol"].unique()
    if len(protocols) < 2:
        print("  Only one protocol found — cannot compare.")
        return

    pivot = df.pivot_table(values="auroc", index="se_name",
                           columns="protocol", aggfunc="mean").round(4)

    orig_cols = [c for c in pivot.columns if "original" in c.lower()]
    fair_cols = [c for c in pivot.columns if "fair" in c.lower()]

    if orig_cols and fair_cols:
        pivot["inflation"] = (pivot[orig_cols[0]] - pivot[fair_cols[0]]).round(4)
        print(f"\n=== {paper_name} — Protocol comparison (AUROC) ===")
        print(pivot.to_string())
        print(f"\n  Mean inflation (original − fair): "
              f"{pivot['inflation'].mean():.4f}")
        print(f"  Max inflation:  {pivot['inflation'].max():.4f}")
        print(f"  Min inflation:  {pivot['inflation'].min():.4f}")
