"""
models/reproductions/nnps_netprop_nmf.py
=========================================
NNPS variant: Network Propagation + NMF features (fair protocol).

Replaces PCA with two biologically grounded steps:

  1. Random-Walk-with-Restart (RWR) propagates each drug's binary target
     signal through the PPI network. Gene j receives signal from gene i
     whenever (i, j) is a PPI edge — capturing guilt-by-association:
     a drug that targets EGFR also implicitly influences ERBB2 neighbours.

  2. NMF (Non-negative Matrix Factorization) replaces PCA:
     - Input is non-negative after propagation → components are additive.
     - Each latent dimension is an interpretable "biological module":
       co-active gene neighbourhoods + mono side-effect patterns that
       load together. Inspect nmf.components_[k] to name each module.
     - No indefinite sign ambiguity (unlike PCA on binary input).

All other protocol choices match the fair baseline (nnps_fair.py):
  - NMF fitted on training drugs only (no transductive leakage)
  - Drug cold-start split
  - Structured negatives
  - hadamard_absdiff pair operator
  - LightGBM classifier (or --use-mlp for the paper's MLP)

Ablation flags
--------------
  --alpha FLOAT       RWR restart prob (default 0.85; 0 = no propagation)
  --n-components INT  NMF rank (default 128)
  --skip-prop         Skip PPI propagation; NMF on raw binary features only
  --random-split      Random pair split instead of cold-start
  --random-neg        Random instead of structured negatives
  --use-mlp           MLP (300→200→100) instead of LightGBM

Usage
-----
    python models/reproductions/nnps_netprop_nmf.py
    python models/reproductions/nnps_netprop_nmf.py --ses all
    python models/reproductions/nnps_netprop_nmf.py --skip-prop
    python models/reproductions/nnps_netprop_nmf.py --alpha 0.5
    python models/reproductions/nnps_netprop_nmf.py --n-components 64
    python models/reproductions/nnps_netprop_nmf.py --use-mlp
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from sklearn.decomposition import NMF

import lightgbm as lgb

from models.reproductions._utils import (
    load_decagon_data, build_negatives, build_splits,
    save_results, SE_NAMES,
)
from models.reproductions.nnps_original import NNPSMlp, train_mlp_per_se
from features.pair_operators import make_pair_features
from evaluation.metrics import compute_metrics
from configs.config import RANDOM_SEED


# ── PPI propagation ────────────────────────────────────────────────────────────

def build_symmetric_adjacency(ppi: pd.DataFrame, all_genes: list):
    """
    Build a symmetrically-normalised PPI adjacency matrix (sparse).

    Uses the normalisation  W = D^{-1/2} A D^{-1/2}  so that eigenvalues
    lie in [-1, 1] and the RWR iteration is guaranteed to converge.
    Self-loops are added before normalisation to prevent isolated-node issues.

    Returns
    -------
    W_sym : (n_genes, n_genes) scipy CSR sparse float32
    g2i   : dict {gene_name: column_index}
    """
    g2i = {g: i for i, g in enumerate(all_genes)}
    n = len(all_genes)

    rows, cols = [], []
    for _, row in ppi.iterrows():
        g1, g2 = row["Gene 1"], row["Gene 2"]
        if g1 in g2i and g2 in g2i:
            i, j = g2i[g1], g2i[g2]
            rows.extend([i, j])
            cols.extend([j, i])

    # Self-loops stabilise propagation for isolated or low-degree genes
    rows.extend(range(n))
    cols.extend(range(n))

    data = np.ones(len(rows), dtype=np.float32)
    A = csr_matrix((data, (rows, cols)), shape=(n, n))

    degree = np.array(A.sum(axis=1)).flatten()
    inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0).astype(np.float32)
    D_inv_sqrt = diags(inv_sqrt)
    W_sym = (D_inv_sqrt @ A @ D_inv_sqrt).astype(np.float32)
    return W_sym, g2i


def random_walk_restart(
    X_target: np.ndarray,
    W_sym,
    alpha: float = 0.85,
    max_iter: int = 30,
    tol: float = 1e-4,
) -> np.ndarray:
    """
    Propagate drug-gene signals through PPI via Random Walk with Restart.

    Iteration:  F_{t+1} = (1 - alpha) * X + alpha * (W_sym @ F_t^T)^T

    alpha  : restart probability (fraction preserved from the initial seed).
             High alpha (0.85) = strong propagation; 0 = no propagation.
    max_iter: safety cap; convergence is typically reached in <15 iterations.
    tol    : L-inf change between iterations to declare convergence.

    Parameters
    ----------
    X_target : (n_drugs, n_genes) non-negative binary drug-protein matrix
    W_sym    : (n_genes, n_genes) symmetrically-normalised sparse adjacency

    Returns
    -------
    F : (n_drugs, n_genes) float32 — propagated drug-gene affinity scores
    """
    F = X_target.astype(np.float32)
    X0 = X_target.astype(np.float32)

    print(f"  RWR propagation (alpha={alpha}, max_iter={max_iter})...")
    for it in range(max_iter):
        # Propagate: for each drug, average its neighbours' scores.
        # W_sym @ F.T has shape (n_genes, n_drugs); transpose back.
        F_propagated = (W_sym @ F.T).T.astype(np.float32)
        F_new = (1.0 - alpha) * X0 + alpha * F_propagated
        delta = float(np.abs(F_new - F).max())
        F = F_new
        if delta < tol:
            print(f"  Converged at iteration {it + 1}  (Δ_max = {delta:.2e})")
            break
    else:
        print(f"  Did not converge after {max_iter} iterations  (Δ_max = {delta:.2e})")

    return F


# ── Feature matrices ───────────────────────────────────────────────────────────

def build_target_matrix(targets: pd.DataFrame, ppi: pd.DataFrame,
                        drugs: list) -> tuple[np.ndarray, list]:
    """Binary drug-protein matrix (n_drugs × n_ppi_genes)."""
    all_genes = sorted(set(ppi["Gene 1"]).union(set(ppi["Gene 2"])))
    g2i = {g: i for i, g in enumerate(all_genes)}
    d2i = {d: i for i, d in enumerate(drugs)}
    drug2genes = targets.groupby("STITCH")["Gene"].apply(set).to_dict()

    X = np.zeros((len(drugs), len(all_genes)), dtype=np.float32)
    for drug in drugs:
        for g in drug2genes.get(drug, set()):
            if g in g2i:
                X[d2i[drug], g2i[g]] = 1.0
    return X, all_genes


def build_mono_matrix(mono: pd.DataFrame, drugs: list) -> np.ndarray:
    """Binary drug-mono-SE matrix (n_drugs × n_mono_SEs)."""
    all_ses = sorted(mono["Individual Side Effect"].unique())
    se2i = {s: i for i, s in enumerate(all_ses)}
    d2i = {d: i for i, d in enumerate(drugs)}
    drug2mono = mono.groupby("STITCH")["Individual Side Effect"].apply(set).to_dict()

    X = np.zeros((len(drugs), len(all_ses)), dtype=np.float32)
    for drug in drugs:
        for se in drug2mono.get(drug, set()):
            if se in se2i:
                X[d2i[drug], se2i[se]] = 1.0
    return X


def fit_nmf_train_only(
    X_combined: np.ndarray,
    train_idx: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """
    Fit NMF on training drug rows only, then transform all drugs.

    This prevents transductive leakage: the latent module basis is
    estimated solely from training drug biology, then projected onto
    val/test drugs. Unlike PCA, NMF components are non-negative and
    additive — each component is a co-activation 'biological program'.

    Returns Z : (n_drugs, n_components) non-negative float32
    """
    n_components = min(n_components, len(train_idx) - 1)
    nmf = NMF(
        n_components=n_components,
        init="nndsvda",       # deterministic warm start, better than random
        max_iter=400,
        random_state=RANDOM_SEED,
        tol=1e-4,
    )
    nmf.fit(X_combined[train_idx])
    Z = nmf.transform(X_combined).astype(np.float32)
    return Z, n_components


# ── Main ──────────────────────────────────────────────────────────────────────

def run_netprop_nmf(
    se_ids=None,
    alpha: float = 0.85,
    n_components: int = 128,
    skip_prop: bool = False,
    random_split: bool = False,
    random_neg: bool = False,
    use_mlp: bool = False,
):
    prop_label = "no_prop" if skip_prop else f"rwr_a{alpha:.2f}"
    split_label = "random_pair" if random_split else "drug_cold_start"
    neg_label   = "random" if random_neg else "structured"
    clf_label   = "mlp" if use_mlp else "lgbm"
    variant     = f"netprop_nmf_{prop_label}_k{n_components}_{clf_label}"

    print("\n" + "=" * 65)
    print("NNPS — Network Propagation + NMF Features (fair protocol)")
    print(f"  Propagation : {'DISABLED (raw binary)' if skip_prop else f'RWR alpha={alpha}'}")
    print(f"  Reduction   : NMF (k={n_components})")
    print(f"  Split       : {split_label}")
    print(f"  Negatives   : {neg_label}")
    print(f"  Classifier  : {'MLP (300→200→100)' if use_mlp else 'LightGBM'}")
    print("=" * 65 + "\n")

    combo_f, mono, ppi, targets, all_drugs, se_ids = load_decagon_data(se_ids)
    d2i = {d: i for i, d in enumerate(all_drugs)}

    # ── Step 1: build raw drug-protein and drug-mono matrices ─────────────────
    print("Building feature matrices...")
    X_target, all_genes = build_target_matrix(targets, ppi, all_drugs)
    X_mono = build_mono_matrix(mono, all_drugs)
    print(f"  Target matrix : {X_target.shape}  "
          f"(density={X_target.mean():.4f})")
    print(f"  Mono   matrix : {X_mono.shape}  "
          f"(density={X_mono.mean():.4f})")

    # ── Step 2: PPI propagation ────────────────────────────────────────────────
    if skip_prop:
        X_propagated = X_target
        print("  Skipping PPI propagation (--skip-prop).")
    else:
        W_sym, _ = build_symmetric_adjacency(ppi, all_genes)
        X_propagated = random_walk_restart(X_target, W_sym, alpha=alpha)
        print(f"  Propagated target matrix: {X_propagated.shape}  "
              f"(mean={X_propagated.mean():.4f}, "
              f"max={X_propagated.max():.4f})")

    # ── Step 3: combine propagated targets + mono SEs ─────────────────────────
    # Mono SEs are already biologically meaningful binary features;
    # we concatenate them with the propagated gene scores before NMF
    # so the model can learn cross-modal latent modules.
    X_combined = np.concatenate([X_propagated, X_mono], axis=1)
    print(f"  Combined matrix : {X_combined.shape}")

    # ── Step 4: negatives & splits ────────────────────────────────────────────
    negatives = build_negatives(combo_f, neg_label, se_ids)
    splits    = build_splits(combo_f, negatives, split_label, se_ids)

    # ── Step 5: per-SE NMF + classifier ───────────────────────────────────────
    lgbm_params = {
        "learning_rate"    : 0.05,
        "num_leaves"       : 63,
        "min_child_samples": 20,
        "subsample"        : 0.8,
        "colsample_bytree" : 0.8,
        "lambda_l1"        : 1.0,
        "n_jobs"           : -1,
        "verbose"          : -1,
    }

    results_rows = []
    print(f"\nPer-SE loop over {len(se_ids)} side effects...")

    for i, se_id in enumerate(se_ids):
        split = splits[se_id]
        if len(split.train) == 0 or split.train["label"].nunique() < 2:
            continue

        # Indices of drugs present in training set
        train_drugs = set(split.train["drug_1"]).union(set(split.train["drug_2"]))
        train_idx   = np.array([d2i[d] for d in all_drugs if d in train_drugs])

        # NMF fitted on training drugs only — no leakage into val/test
        Z, actual_k = fit_nmf_train_only(X_combined, train_idx, n_components)
        drug2idx = {d: idx for idx, d in enumerate(all_drugs)}

        def get_X(df: pd.DataFrame) -> np.ndarray:
            zi = Z[np.array([drug2idx[d] for d in df["drug_1"]])]
            zj = Z[np.array([drug2idx[d] for d in df["drug_2"]])]
            return make_pair_features(zi, zj, operator="hadamard_absdiff")

        X_tr = get_X(split.train); y_tr = split.train["label"].values
        X_vl = get_X(split.val);   y_vl = split.val["label"].values
        X_te = get_X(split.test);  y_te = split.test["label"].values

        if use_mlp:
            proba = train_mlp_per_se(
                X_tr, y_tr, X_vl, y_vl, X_te, y_te,
                input_dim=X_tr.shape[1],
            )
        else:
            n_pos = y_tr.sum()
            n_neg = len(y_tr) - n_pos
            n_train = len(y_tr)
            num_leaves = max(7, min(63, n_train // 50))
            params = {
                **lgbm_params,
                "num_leaves"       : num_leaves,
                "scale_pos_weight" : n_neg / max(n_pos, 1),
            }
            dtrain  = lgb.Dataset(X_tr, label=y_tr)
            dval    = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)
            booster = lgb.train(
                params, dtrain, num_boost_round=1000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
            proba = booster.predict(X_te)

        metrics = compute_metrics(y_te, proba)
        model_label = f"netprop_nmf_{'mlp' if use_mlp else 'lgbm'}"

        results_rows.append({
            "se_id"       : se_id,
            "se_name"     : SE_NAMES.get(se_id, se_id),
            "model"       : model_label,
            "protocol"    : variant,
            "split"       : split_label,
            "neg"         : neg_label,
            "operator"    : "hadamard_absdiff",
            "features"    : f"{'rwr' if not skip_prop else 'raw'}_target+mono_nmf{actual_k}d",
            "alpha"       : alpha if not skip_prop else 0.0,
            "n_components": actual_k,
            **metrics,
        })

        if (i + 1) % 3 == 0 or i == 0:
            print(f"  [{i+1:2d}/{len(se_ids)}] {SE_NAMES.get(se_id, se_id):35s} "
                  f"AUROC={metrics['auroc']:.4f}  AP={metrics['ap']:.4f}  "
                  f"k={actual_k}")

    results = pd.DataFrame(results_rows)
    print(f"\n  Mean AUROC ({variant}): {results['auroc'].mean():.4f}")
    print(f"  Mean AP   ({variant}): {results['ap'].mean():.4f}")
    save_results(results, f"nnps_{variant}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NNPS with Network Propagation + NMF features (fair protocol)"
    )
    parser.add_argument("--ses", default="rep",
                        help="'rep' for 12 representative SEs, 'all' for all 963")
    parser.add_argument("--alpha", type=float, default=0.85,
                        help="RWR restart probability (default: 0.85)")
    parser.add_argument("--n-components", type=int, default=128,
                        help="NMF rank / number of latent biological modules (default: 128)")
    parser.add_argument("--skip-prop", action="store_true",
                        help="Skip PPI propagation; apply NMF to raw binary features")
    parser.add_argument("--random-split", action="store_true",
                        help="Use random pair split instead of cold-start (introduces leakage)")
    parser.add_argument("--random-neg", action="store_true",
                        help="Use random negatives instead of structured")
    parser.add_argument("--use-mlp", action="store_true",
                        help="Use MLP (300→200→100) instead of LightGBM")
    args = parser.parse_args()

    se_ids = None if args.ses == "rep" else "all"
    run_netprop_nmf(
        se_ids=se_ids,
        alpha=args.alpha,
        n_components=args.n_components,
        skip_prop=args.skip_prop,
        random_split=args.random_split,
        random_neg=args.random_neg,
        use_mlp=args.use_mlp,
    )
