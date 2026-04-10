"""
models/reproductions/nnps_fair.py
==================================
Reproduction of NNPS (Masumshah et al., 2021) under the FAIR protocol.

Each methodological flaw from the original is corrected:
  1. TruncatedSVD fitted on TRAINING drugs only (no PCA leakage)
  2. Drug-level cold-start split (no drug-overlap leakage)
  3. Structured negatives (harder, more realistic)
  4. hadamard_absdiff pair operator (symmetric, no identity leakage)
  5. Separate target and mono embeddings via GDSVD + JaccardSVD

Every fix is individually toggleable to isolate contributions:
  --pca-leakage      : reintroduce PCA fitted on all drugs
  --random-split     : use random pair split instead of cold-start
  --random-neg       : use random instead of structured negatives
  --sum-operator     : use sum instead of hadamard_absdiff
  --use-mlp          : use MLP (300→200→100) instead of LightGBM

Usage
-----
    python models/reproductions/nnps_fair.py
    python models/reproductions/nnps_fair.py --ses all
    python models/reproductions/nnps_fair.py --pca-leakage   # isolate fix 1
    python models/reproductions/nnps_fair.py --random-split  # isolate fix 2
    python models/reproductions/nnps_fair.py --use-mlp       # paper architecture
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix

from models.reproductions._utils import (
    load_decagon_data, build_negatives, build_splits,
    save_results, compare_protocols, SE_NAMES,
)
from features.pair_operators import make_pair_features
from evaluation.metrics import compute_metrics
from configs.config import METRICS_DIR, CACHE_DIR, RANDOM_SEED

import lightgbm as lgb
from models.reproductions.nnps_original import NNPSMlp, train_mlp_per_se


# ── Fair feature construction ─────────────────────────────────────────────────

def build_target_matrix(targets, ppi, drugs):
    """Binary drug-protein matrix (n_drugs × n_genes)."""
    all_genes = sorted(set(ppi["Gene 1"]).union(set(ppi["Gene 2"])))
    g2i = {g: i for i, g in enumerate(all_genes)}
    d2i = {d: i for i, d in enumerate(drugs)}
    drug2genes = targets.groupby("STITCH")["Gene"].apply(set).to_dict()
    X = np.zeros((len(drugs), len(all_genes)), dtype=np.float32)
    for drug in drugs:
        for g in drug2genes.get(drug, set()):
            if g in g2i:
                X[d2i[drug], g2i[g]] = 1.0
    return X


def build_mono_matrix(mono, drugs):
    """Binary drug-mono-SE matrix (n_drugs × n_mono_SEs)."""
    all_ses = sorted(mono["Individual Side Effect"].unique())
    se2i = {s: i for i, s in enumerate(all_ses)}
    d2i  = {d: i for i, d in enumerate(drugs)}
    drug2mono = mono.groupby("STITCH")["Individual Side Effect"].apply(set).to_dict()
    X = np.zeros((len(drugs), len(all_ses)), dtype=np.float32)
    for drug in drugs:
        for se in drug2mono.get(drug, set()):
            if se in se2i:
                X[d2i[drug], se2i[se]] = 1.0
    return X


def build_jaccard_matrix(X_mono):
    """Drug-drug Jaccard similarity from mono SE profiles (n_drugs × n_drugs)."""
    norms = X_mono.sum(axis=1)
    dot   = X_mono @ X_mono.T
    union = norms[:, None] + norms[None, :] - dot
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(union > 0, dot / union, 0.0)
    np.fill_diagonal(J, 1.0)
    return J.astype(np.float32)


def fit_truncsvd_train_only(X_train_drugs, X_all_drugs, n_components,
                             use_pca_leakage=False):
    """
    Fit TruncatedSVD on training drugs only, transform all drugs.

    If use_pca_leakage=True, fits on all drugs (replicates NNPS flaw).
    """
    if use_pca_leakage:
        print(f"    [LEAKAGE MODE] SVD fitted on ALL {len(X_all_drugs)} drugs")
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        svd.fit(csr_matrix(X_all_drugs))
    else:
        print(f"    [FAIR MODE] SVD fitted on {len(X_train_drugs)} training drugs only")
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
        svd.fit(csr_matrix(X_train_drugs))
    return svd.transform(csr_matrix(X_all_drugs)).astype(np.float32)


class NNPSFairFeatureSet:
    """
    Feature set compatible with LGBMPredictor.get() interface.
    Holds separate target and mono embeddings for each drug.
    """
    def __init__(self, drugs, Z_target, Z_mono):
        self.drug2idx = {d: i for i, d in enumerate(drugs)}
        self.Z_target = Z_target   # (n, target_dim)
        self.Z_mono   = Z_mono     # (n, mono_dim)

    def get(self, drug_ids, mode="target+mono"):
        idx = np.array([self.drug2idx[d] for d in drug_ids])
        if mode == "target_only":
            return self.Z_target[idx]
        elif mode == "mono_only":
            return self.Z_mono[idx]
        else:  # target+mono (default)
            return np.concatenate([self.Z_target[idx], self.Z_mono[idx]], axis=1)

    def feature_dim(self, mode="target+mono"):
        d = self.get(list(self.drug2idx)[:1], mode=mode)
        return d.shape[1]


# ── Main ──────────────────────────────────────────────────────────────────────

def run_nnps_fair(se_ids=None, pca_leakage=False, random_split=False,
                  random_neg=False, sum_operator=False, use_mlp=False):

    label = "fair" + ("_pcaleak" if pca_leakage else "") + \
            ("_randsplit" if random_split else "") + \
            ("_randneg" if random_neg else "") + \
            ("_sum" if sum_operator else "") + \
            ("_mlp" if use_mlp else "")

    classifier = "MLP (300→200→100)" if use_mlp else "LightGBM"
    print("\n" + "="*65)
    print(f"NNPS FAIR PROTOCOL — variant: {label}")
    print("Fixes: no PCA leakage | cold-start split | structured neg | hadamard_absdiff")
    print(f"Classifier: {classifier}")
    print("="*65 + "\n")

    combo_f, mono, ppi, targets, all_drugs, se_ids = load_decagon_data(se_ids)
    d2i = {d: i for i, d in enumerate(all_drugs)}

    # Build raw feature matrices
    X_target_all = build_target_matrix(targets, ppi, all_drugs)
    X_mono_all   = build_mono_matrix(mono, all_drugs)

    # Negatives
    neg_strategy = "random" if random_neg else "structured"
    negatives = build_negatives(combo_f, neg_strategy, se_ids)

    # Split
    split_strategy = "random_pair" if random_split else "drug_cold_start"
    splits = build_splits(combo_f, negatives, split_strategy, se_ids)

    operator = "sum" if sum_operator else "hadamard_absdiff"

    # Per-SE embedding: fit SVD on training drugs only
    results_rows = []
    lgbm_params = {
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda_l1": 1.0,
        "n_jobs": -1,
        "verbose": -1,
    }

    classifier = "MLP (300→200→100)" if use_mlp else "LightGBM"
    print(f"\nTraining per-SE {classifier} ({len(se_ids)} SEs)...")
    for i, se_id in enumerate(se_ids):
        split = splits[se_id]
        if len(split.train) == 0 or split.train["label"].nunique() < 2:
            continue

        # Get training drug indices for fair SVD fitting
        train_drugs = set(split.train["drug_1"]).union(set(split.train["drug_2"]))
        train_idx   = np.array([d2i[d] for d in all_drugs if d in train_drugs])

        # Adaptive num_leaves
        n_train = len(split.train)
        num_leaves = max(7, min(63, n_train // 50))

        # Fit embeddings on training drugs only (or all if leakage mode)
        n_target_dim = min(64, len(train_idx) - 1)
        n_mono_dim   = min(64, len(train_idx) - 1)

        Z_target = fit_truncsvd_train_only(
            X_target_all[train_idx], X_target_all, n_target_dim, pca_leakage)
        Z_mono   = fit_truncsvd_train_only(
            X_mono_all[train_idx],   X_mono_all,   n_mono_dim,   pca_leakage)

        fset = NNPSFairFeatureSet(all_drugs, Z_target, Z_mono)

        def get_X(df):
            zi = fset.get(df["drug_1"].tolist())
            zj = fset.get(df["drug_2"].tolist())
            return make_pair_features(zi, zj, operator=operator)

        X_tr = get_X(split.train); y_tr = split.train["label"].values
        X_vl = get_X(split.val);   y_vl = split.val["label"].values
        X_te = get_X(split.test);  y_te = split.test["label"].values

        if use_mlp:
            input_dim = X_tr.shape[1]
            proba = train_mlp_per_se(
                X_tr, y_tr, X_vl, y_vl, X_te, y_te,
                input_dim=input_dim,
            )
        else:
            n_pos = y_tr.sum(); n_neg = len(y_tr) - n_pos
            params = {**lgbm_params,
                      "num_leaves": num_leaves,
                      "scale_pos_weight": n_neg / max(n_pos, 1)}
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval   = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)
            booster = lgb.train(
                params, dtrain, num_boost_round=1000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
            proba = booster.predict(X_te)

        metrics = compute_metrics(y_te, proba)
        model_label = "nnps_mlp" if use_mlp else "nnps_lgbm"

        results_rows.append({
            "se_id"   : se_id,
            "se_name" : SE_NAMES.get(se_id, se_id),
            "model"   : model_label,
            "protocol": label,
            "split"   : split_strategy,
            "neg"     : neg_strategy,
            "operator": operator,
            "features": f"trunc_svd_target{n_target_dim}+mono{n_mono_dim}",
            "num_leaves": num_leaves,
            **metrics,
        })

        if (i + 1) % 3 == 0 or i == 0:
            extra = f"leaves={num_leaves}" if not use_mlp else f"dim={X_tr.shape[1]}"
            print(f"  [{i+1:2d}/{len(se_ids)}] {SE_NAMES.get(se_id, se_id):35s} "
                  f"AUROC={metrics['auroc']:.4f}  {extra}")

    results = pd.DataFrame(results_rows)
    print(f"\n  Mean AUROC ({label}): {results['auroc'].mean():.4f}")
    print(f"  Mean AP   ({label}): {results['ap'].mean():.4f}")
    save_results(results, f"nnps_{label}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NNPS reproduction under fair or original protocol")
    parser.add_argument("--ses",           default="rep")
    parser.add_argument("--pca-leakage",   action="store_true")
    parser.add_argument("--random-split",  action="store_true")
    parser.add_argument("--random-neg",    action="store_true")
    parser.add_argument("--sum-operator",  action="store_true")
    parser.add_argument("--use-mlp",       action="store_true",
                        help="Use MLP (300→200→100) instead of LightGBM")
    parser.add_argument("--run-both",      action="store_true",
                        help="Run original + fair and compare")
    args = parser.parse_args()

    se_ids = None if args.ses == "rep" else "all"

    if args.run_both:
        from models.reproductions.nnps_original import run_nnps_original
        r_orig = run_nnps_original(se_ids=se_ids, use_lgbm=not args.use_mlp)
        r_orig["protocol"] = "original (inflated)"
        r_fair = run_nnps_fair(se_ids=se_ids, use_mlp=args.use_mlp)
        r_fair["protocol"] = "fair"
        combined = pd.concat([r_orig, r_fair], ignore_index=True)
        compare_protocols(combined, "NNPS")
        save_results(combined, "nnps_combined")
    else:
        run_nnps_fair(
            se_ids=se_ids,
            pca_leakage=args.pca_leakage,
            random_split=args.random_split,
            random_neg=args.random_neg,
            sum_operator=args.sum_operator,
            use_mlp=args.use_mlp,
        )
