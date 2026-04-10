"""
models/reproductions/nnps_netprop_nmf.py
=========================================
NNPS variant: Network Propagation + NMF features (fair protocol).

Replaces PCA with three biologically grounded feature blocks fed into NMF:

  1. RWR-propagated target signal  (n_drugs × n_ppi_genes)
     Random-Walk-with-Restart spreads each drug's protein-target hits
     through the PPI network — guilt-by-association.
     Drugs with NO known targets get an all-zero row here.

  2. Mono side-effect profile  (n_drugs × n_mono_SEs)
     Binary: known single-drug phenotypic side effects.
     Complementary to targets; available for most drugs.

  3. Morgan fingerprints + physicochemical descriptors  (n_drugs × 2055)
     ECFP4 bits (2048) + MW, LogP, HBD, HBA, TPSA, RotBonds, Rings.
     Derived from SMILES — purely chemical, NO target knowledge required.
     THIS IS THE KEY FALLBACK FOR TARGET-MISSING DRUGS (~53% of drugs
     and ~24% of pairs across all SEs have no protein target data).

All three blocks are max-normalised to [0,1] before concatenation so NMF
treats each block with equal weight (the propagated target values are
tiny floats vs. binary {0,1} in the other blocks without normalisation).

NMF then learns k latent "biological modules" spanning all three modalities:
chemical scaffold × PPI neighbourhood × phenotypic side-effect pattern.

All other protocol choices match the fair baseline (nnps_fair.py):
  - NMF fitted on training drugs only (no transductive leakage)
  - Drug cold-start split
  - Structured negatives
  - hadamard_absdiff pair operator
  - LightGBM classifier (or --use-mlp for the paper's MLP)

Dataset context (from per_se_missing_targets.csv)
--------------------------------------------------
  ~53% of drugs across all SEs have NO protein target annotation.
  ~24% of drug pairs have BOTH drugs missing targets.
  Without fingerprints these pairs have zero RWR signal; with fingerprints
  the chemical structure still provides a non-zero embedding.

  Representative SE training sizes (positives + 1:1 negatives, 70/10/20):
    Pulmonary embolism      train~17042  val~2434  test~4869  miss=56%
    Hyperlipaemia           train~12842  val~1834  test~3669  miss=54%
    Drug addiction          train~10893  val~1556  test~3112  miss=50%
    Agranulocytosis         train~ 5990  val~ 855  test~1711  miss=55%
    Herpes simplex          train~ 6076  val~ 868  test~1736  miss=54%
    Micturition urgency     train~ 4933  val~ 704  test~1409  miss=50%
    Meningitis              train~ 3193  val~ 456  test~ 912  miss=56%
    Superficial thrombo.    train~ 2763  val~ 394  test~ 789  miss=52%
    External ear infection  train~ 2576  val~ 368  test~ 736  miss=49%
    Intraocular inflam.     train~ 1201  val~ 171  test~ 343  miss=52%
    Burns second degree     train~ 1072  val~ 153  test~ 306  miss=46%
    Viral encephalitis      train~ 1058  val~ 151  test~ 302  miss=55%

Ablation flags
--------------
  --alpha FLOAT       RWR restart prob (default 0.85; 0 = no propagation)
  --n-components INT  NMF rank (default 128)
  --skip-prop         Skip PPI propagation; NMF on raw binary features only
  --no-fp             Exclude fingerprints (reverts to target+mono only)
  --random-split      Random pair split instead of cold-start
  --random-neg        Random instead of structured negatives
  --use-mlp           MLP (300→200→100) instead of LightGBM

Usage
-----
    python models/reproductions/nnps_netprop_nmf.py
    python models/reproductions/nnps_netprop_nmf.py --ses all
    python models/reproductions/nnps_netprop_nmf.py --no-fp       # ablate fingerprints
    python models/reproductions/nnps_netprop_nmf.py --skip-prop   # ablate propagation
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
from configs.config import RANDOM_SEED, CACHE_DIR

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("[nnps_netprop_nmf] RDKit not available — fingerprints disabled.")


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


def build_fingerprint_matrix(
    drugs: list,
    smiles_cache_path: str,
    morgan_radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """
    ECFP4 Morgan fingerprints (n_bits) + 7 physicochemical descriptors
    for each drug in `drugs`. Drugs with no SMILES or unparseable molecules
    receive an all-zero row — identical fallback to missing-target drugs in
    X_target. The physchem descriptors are divided by fixed scales so they
    sit in roughly [0, 1] like the binary fingerprint bits.

    This block is the only non-zero signal for ~53% of drugs that have no
    protein target annotations. Without it, those drugs are invisible to the
    NMF regardless of how well the PPI propagation works.

    Returns
    -------
    X_fp : (n_drugs, n_bits + 7) float32
    n_ok : int — number of drugs that have valid SMILES
    """
    n_physchem = 7
    X = np.zeros((len(drugs), n_bits + n_physchem), dtype=np.float32)

    if not RDKIT_OK:
        print("  [WARNING] RDKit not available — fingerprint block is all-zero.")
        return X, 0

    smiles_dict: dict = {}
    if os.path.exists(smiles_cache_path):
        sc = pd.read_csv(smiles_cache_path)
        smiles_dict = {r["STITCH"]: r["SMILES"]
                       for _, r in sc.iterrows() if pd.notna(r["SMILES"])}
        print(f"  Loaded {len(smiles_dict)} SMILES from cache.")
    else:
        print(f"  [WARNING] SMILES cache not found at {smiles_cache_path}.")

    gen = GetMorganGenerator(radius=morgan_radius, fpSize=n_bits)
    physchem_fns   = [Descriptors.MolWt, Descriptors.MolLogP,
                      rdMolDescriptors.CalcNumHBD, rdMolDescriptors.CalcNumHBA,
                      Descriptors.TPSA, rdMolDescriptors.CalcNumRotatableBonds,
                      rdMolDescriptors.CalcNumRings]
    physchem_scales = [500.0, 5.0, 10.0, 10.0, 150.0, 10.0, 6.0]

    n_ok = 0
    for i, drug in enumerate(drugs):
        smi = smiles_dict.get(drug)
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        X[i, :n_bits] = (
            np.frombuffer(fp.ToBitString().encode(), dtype=np.uint8) - ord("0")
        )
        for j, (fn, scale) in enumerate(zip(physchem_fns, physchem_scales)):
                try:
                    # Clip to 0: NMF requires non-negative input.
                    # LogP in particular can be negative for hydrophilic drugs.
                    X[i, n_bits + j] = max(0.0, float(fn(mol)) / scale)
                except Exception:
                    pass
        n_ok += 1

    n_miss = len(drugs) - n_ok
    print(f"  Fingerprint matrix : {X.shape}  "
          f"({n_ok} drugs with SMILES, {n_miss} without → zero rows)")
    return X, n_ok


def max_normalise(X: np.ndarray, name: str) -> np.ndarray:
    """
    Scale a feature block to [0, 1] by dividing by its global maximum.
    Binary matrices (mono, fingerprints) are unchanged (max=1).
    The RWR-propagated target block has small float values that would
    otherwise be dominated by the binary blocks in the NMF objective.
    """
    m = X.max()
    if m > 0:
        return (X / m).astype(np.float32)
    print(f"  [WARNING] Block '{name}' is all-zero after propagation.")
    return X


def fit_nmf_train_only(
    X_combined: np.ndarray,
    train_idx: np.ndarray,
    n_components: int,
    n_train_samples: int,
    min_ratio: float = 5.0,
) -> tuple[np.ndarray, int]:
    """
    Fit NMF on training drug rows only, then transform all drugs.

    This prevents transductive leakage: the latent module basis is
    estimated solely from training drug biology, then projected onto
    val/test drugs. Unlike PCA, NMF components are non-negative and
    additive — each component is a co-activation 'biological program'.

    Adaptive rank selection
    -----------------------
    The classifier receives hadamard_absdiff pair features of shape (n, 2k).
    To keep the samples-to-features ratio ≥ min_ratio we cap k at:

        k ≤ n_train_samples / (2 * min_ratio)

    Three caps therefore apply (tightest wins):
      1. User-specified --n-components (global ceiling)
      2. n_train_drugs - 1  (NMF cannot have more components than samples)
      3. n_train_samples / (2 * min_ratio)  (overfitting guard)

    With min_ratio=5, viral encephalitis (n≈740) gets k≤74 instead of 113,
    keeping the feature-to-sample ratio safe for LightGBM.

    Parameters
    ----------
    X_combined      : (n_drugs, n_features) combined feature block
    train_idx       : indices of training drugs in all_drugs
    n_components    : maximum requested NMF rank (--n-components flag)
    n_train_samples : number of training pairs (positives + negatives)
    min_ratio       : minimum required n_train_samples / (2k) ratio

    Returns
    -------
    Z        : (n_drugs, actual_k) non-negative float32 NMF embeddings
    actual_k : effective rank used (may be < n_components)
    """
    # Cap 1: user ceiling
    k = n_components
    # Cap 2: NMF cannot exceed number of training drug rows
    k = min(k, len(train_idx) - 1)
    # Cap 3: samples-to-features guard (pair features = 2k)
    k_safe = max(1, int(n_train_samples / (2.0 * min_ratio)))
    if k_safe < k:
        print(f"    [adaptive k] {k} → {k_safe}  "
              f"(n_train={n_train_samples}, ratio would be "
              f"{n_train_samples / (2.0 * k):.1f} < {min_ratio:.0f})")
        k = k_safe

    nmf = NMF(
        n_components=k,
        init="nndsvda",   # deterministic SVD warm start, avoids random local minima
        max_iter=400,
        random_state=RANDOM_SEED,
        tol=1e-4,
    )
    # NMF requires non-negative input; clip guards against any residual
    # negative values (e.g. negative LogP after scale division).
    X_nn = np.clip(X_combined, 0, None)
    nmf.fit(X_nn[train_idx])
    Z = nmf.transform(X_nn).astype(np.float32)
    return Z, k


# ── Main ──────────────────────────────────────────────────────────────────────

def run_netprop_nmf(
    se_ids=None,
    alpha: float = 0.85,
    n_components: int = 128,
    skip_prop: bool = False,
    use_fp: bool = True,
    random_split: bool = False,
    random_neg: bool = False,
    use_mlp: bool = False,
):
    prop_label  = "no_prop" if skip_prop else f"rwr_a{alpha:.2f}"
    fp_label    = "fp" if use_fp else "nofp"
    split_label = "random_pair" if random_split else "drug_cold_start"
    neg_label   = "random" if random_neg else "structured"
    clf_label   = "mlp" if use_mlp else "lgbm"
    variant     = f"netprop_nmf_{prop_label}_{fp_label}_k{n_components}_{clf_label}"

    print("\n" + "=" * 65)
    print("NNPS — Network Propagation + NMF Features (fair protocol)")
    print(f"  Propagation  : {'DISABLED (raw binary)' if skip_prop else f'RWR alpha={alpha}'}")
    print(f"  Fingerprints : {'YES (ECFP4 + physchem, fed into NMF)' if use_fp else 'NO (--no-fp)'}")
    print(f"  Reduction    : NMF (k={n_components}) on target+mono+fp")
    print(f"  Split        : {split_label}")
    print(f"  Negatives    : {neg_label}")
    print(f"  Classifier   : {'MLP (300→200→100)' if use_mlp else 'LightGBM'}")
    print("=" * 65 + "\n")

    combo_f, mono, ppi, targets, all_drugs, se_ids = load_decagon_data(se_ids)
    d2i = {d: i for i, d in enumerate(all_drugs)}

    # ── Step 1: build raw feature matrices ────────────────────────────────────
    print("Building feature matrices...")
    X_target, all_genes = build_target_matrix(targets, ppi, all_drugs)
    X_mono = build_mono_matrix(mono, all_drugs)
    n_no_target = int((X_target.sum(axis=1) == 0).sum())
    print(f"  Target matrix  : {X_target.shape}  "
          f"({n_no_target}/{len(all_drugs)} drugs have no targets → zero rows)")
    print(f"  Mono   matrix  : {X_mono.shape}  "
          f"(density={X_mono.mean():.4f})")

    # ── Step 2: PPI propagation ────────────────────────────────────────────────
    if skip_prop:
        X_propagated = X_target
        print("  Skipping PPI propagation (--skip-prop).")
    else:
        W_sym, _ = build_symmetric_adjacency(ppi, all_genes)
        X_propagated = random_walk_restart(X_target, W_sym, alpha=alpha)
        print(f"  Propagated     : {X_propagated.shape}  "
              f"(mean={X_propagated.mean():.5f}, max={X_propagated.max():.4f})")

    # ── Step 3: Morgan fingerprints ───────────────────────────────────────────
    # Drugs with no protein targets have all-zero rows in X_propagated.
    # Fingerprints provide non-zero signal for those drugs via chemical
    # structure alone, covering ~53% of drugs and ~24% of pairs.
    if use_fp:
        smiles_cache = os.path.join(CACHE_DIR, "smiles.csv")
        X_fp, _ = build_fingerprint_matrix(all_drugs, smiles_cache)
    else:
        X_fp = None

    # ── Step 4: normalise and concatenate all blocks ──────────────────────────
    # Max-normalise each block to [0, 1] so that the RWR propagated values
    # (small floats) are on the same scale as the binary {0,1} blocks.
    # Binary matrices are unchanged by this operation (their max = 1).
    blocks = [
        max_normalise(X_propagated, "propagated_target"),
        max_normalise(X_mono,       "mono_se"),
    ]
    block_desc = "rwr_target+mono"
    if X_fp is not None:
        blocks.append(max_normalise(X_fp, "fingerprints"))
        block_desc += "+fp"

    X_combined = np.concatenate(blocks, axis=1)
    print(f"  Combined matrix : {X_combined.shape}  (blocks: {block_desc})")

    # ── Step 5: negatives & splits ────────────────────────────────────────────
    negatives = build_negatives(combo_f, neg_label, se_ids)
    splits    = build_splits(combo_f, negatives, split_label, se_ids)

    # ── Step 6: per-SE NMF + classifier ───────────────────────────────────────
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

        # Indices of unique drugs present in the training split
        train_drugs = set(split.train["drug_1"]).union(set(split.train["drug_2"]))
        train_idx   = np.array([d2i[d] for d in all_drugs if d in train_drugs])
        n_train     = len(split.train)       # number of training (pair) samples

        # Warn early if the SE is tiny — ratio is computed inside fit_nmf_train_only
        # but surface it here so it's visible in the per-SE progress line.
        raw_ratio = n_train / (2.0 * min(n_components, len(train_idx) - 1))
        if raw_ratio < 5.0:
            print(f"    [WARNING] SE '{SE_NAMES.get(se_id, se_id)}': "
                  f"n_train={n_train}, uncapped pair_feats={2*min(n_components, len(train_idx)-1)}, "
                  f"ratio={raw_ratio:.1f} — k will be reduced to maintain ratio≥5")

        # NMF fitted on training drugs only — no leakage into val/test.
        # Adaptive k ensures n_train / (2k) ≥ 5 (controlled overfitting risk).
        Z, actual_k = fit_nmf_train_only(
            X_combined, train_idx, n_components, n_train_samples=n_train)
        drug2idx = {d: idx for idx, d in enumerate(all_drugs)}

        actual_ratio = n_train / max(2 * actual_k, 1)

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
            # num_leaves: fewer leaves for smaller training sets
            num_leaves = max(7, min(63, n_train // 50))
            # min_child_samples: scale with training size; floor at 5, cap at 20.
            # Prevents leaf splits on <5 samples when n_train is very small.
            min_child = max(5, min(20, n_train // 30))
            params = {
                **lgbm_params,
                "num_leaves"        : num_leaves,
                "min_child_samples" : min_child,
                "scale_pos_weight"  : n_neg / max(n_pos, 1),
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
            "se_id"          : se_id,
            "se_name"        : SE_NAMES.get(se_id, se_id),
            "model"          : model_label,
            "protocol"       : variant,
            "split"          : split_label,
            "neg"            : neg_label,
            "operator"       : "hadamard_absdiff",
            "features"       : f"{block_desc}_nmf{actual_k}d",
            "alpha"          : alpha if not skip_prop else 0.0,
            "n_components"   : actual_k,
            "n_train"        : n_train,
            "pair_feats"     : 2 * actual_k,
            "sample_feat_ratio": round(actual_ratio, 2),
            **metrics,
        })

        if (i + 1) % 3 == 0 or i == 0:
            ratio_flag = " (!)" if actual_ratio < 5 else ""
            print(f"  [{i+1:2d}/{len(se_ids)}] {SE_NAMES.get(se_id, se_id):35s} "
                  f"AUROC={metrics['auroc']:.4f}  AP={metrics['ap']:.4f}  "
                  f"k={actual_k}  ratio={actual_ratio:.1f}{ratio_flag}")

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
    parser.add_argument("--no-fp", action="store_true",
                        help="Exclude Morgan fingerprints (ablate the chemical block)")
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
        use_fp=not args.no_fp,
        random_split=args.random_split,
        random_neg=args.random_neg,
        use_mlp=args.use_mlp,
    )
