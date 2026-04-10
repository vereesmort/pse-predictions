"""
models/reproductions/nnps_original.py
======================================
Reproduction of NNPS (Masumshah et al., 2021) under the ORIGINAL protocol.

This intentionally replicates ALL methodological flaws from the paper:
  1. PCA fitted on ALL 645 drugs before any split (transductive PCA leakage)
  2. Random pair-level 80/10/10 split (drug-overlap leakage)
  3. Random negatives
  4. Sum pair aggregation (f_i + f_j)
  5. Concatenated mono + target features reduced to 525 dims via PCA

Purpose: produce the inflated AUROC ~0.966 to compare directly with
the paper's reported result and then against the fair reproduction.

Model architecture (faithful to Masumshah et al., 2021):
  - Input: 525-dimensional PCA-compressed sum pair vector
  - Hidden layers: 300 → 200 → 100 neurons, ReLU, Dropout(0.1)
  - Output: 1 neuron, Sigmoid
  - Optimizer: SGD(lr=0.01, momentum=0.9)
  - Batch size: 1,024
  - Epochs: 50 (early stopping on validation loss, patience=10)
  - Loss: Binary cross-entropy

For quick comparison, pass --use-lgbm to substitute LightGBM instead.

Usage
-----
    python models/reproductions/nnps_original.py
    python models/reproductions/nnps_original.py --ses all
    python models/reproductions/nnps_original.py --use-lgbm
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.reproductions._utils import (
    load_decagon_data, build_negatives, save_results, SE_NAMES,
)
from evaluation.metrics import compute_metrics
from configs.config import METRICS_DIR, CACHE_DIR, RANDOM_SEED


# ── NNPS feature construction (original protocol) ─────────────────────────────

def build_nnps_features_original(targets, ppi, mono, all_drugs,
                                  n_components=525):
    """
    Replicate NNPS feature construction exactly:
    1. Build drug-protein interaction matrix (645 × n_genes)
    2. Build mono side-effect matrix (645 × 10184)
    3. Concatenate: (645 × (n_genes + 10184))
    4. StandardScaler (no mean, with_std=False as in sparse matrices)
    5. PCA fitted on ALL 645 drugs → 525 dims (95 % variance)

    FLAW: PCA is fitted on the full drug set before any split,
    so test drugs influence the PCA projection space.
    """
    drug2idx = {d: i for i, d in enumerate(all_drugs)}
    n_drugs  = len(all_drugs)

    all_genes = sorted(set(ppi["Gene 1"]).union(set(ppi["Gene 2"])))
    g2i = {g: i for i, g in enumerate(all_genes)}
    drug2genes = targets.groupby("STITCH")["Gene"].apply(set).to_dict()
    X_target = np.zeros((n_drugs, len(all_genes)), dtype=np.float32)
    for drug in all_drugs:
        i = drug2idx[drug]
        for g in drug2genes.get(drug, set()):
            if g in g2i:
                X_target[i, g2i[g]] = 1.0

    all_mono_ses = sorted(mono["Individual Side Effect"].unique())
    se2i = {s: i for i, s in enumerate(all_mono_ses)}
    drug2mono = mono.groupby("STITCH")["Individual Side Effect"].apply(set).to_dict()
    X_mono = np.zeros((n_drugs, len(all_mono_ses)), dtype=np.float32)
    for drug in all_drugs:
        i = drug2idx[drug]
        for se in drug2mono.get(drug, set()):
            if se in se2i:
                X_mono[i, se2i[se]] = 1.0

    X_combined = np.concatenate([X_target, X_mono], axis=1)
    print(f"  Combined matrix: {X_combined.shape}")

    # PCA on ALL drugs — the leakage the paper has
    print(f"  Fitting PCA(n={n_components}) on ALL {n_drugs} drugs (replicating leakage)...")
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X_combined)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    Z = pca.fit_transform(X_scaled)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f} "
          f"({n_components} components)")
    return Z.astype(np.float32), drug2idx


def make_pair_sum(Z, drug2idx, drug_1_list, drug_2_list):
    """Sum pair aggregation: z_i + z_j (NNPS default)."""
    i_idx = np.array([drug2idx[d] for d in drug_1_list])
    j_idx = np.array([drug2idx[d] for d in drug_2_list])
    return (Z[i_idx] + Z[j_idx]).astype(np.float32)


# ── MLP (faithful to Masumshah et al., 2021) ──────────────────────────────────

class NNPSMlp(nn.Module):
    """
    Three-layer feed-forward network from Masumshah et al., 2021.
      300 → 200 → 100 → 1  with ReLU + Dropout(0.1) after each hidden layer.
    """
    def __init__(self, input_dim: int = 525, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 300),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(100, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_mlp_per_se(X_tr, y_tr, X_vl, y_vl, X_te, y_te,
                     input_dim=525, lr=0.01, momentum=0.9,
                     batch_size=1024, max_epochs=50, patience=10,
                     device="cpu"):
    """Train the NNPS MLP, return test probabilities and best val loss."""
    model = NNPSMlp(input_dim=input_dim).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = nn.BCELoss()

    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
    train_loader = DataLoader(
        TensorDataset(t(X_tr), t(y_tr.astype(np.float32))),
        batch_size=batch_size, shuffle=True,
    )
    X_vl_t, y_vl_t = t(X_vl), t(y_vl.astype(np.float32))
    X_te_t = t(X_te)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_vl_t), y_vl_t).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        proba = model(X_te_t).cpu().numpy()
    return proba


# ── Main ──────────────────────────────────────────────────────────────────────

def run_nnps_original(se_ids=None, use_lgbm=False):
    classifier = "LightGBM" if use_lgbm else "MLP (300→200→100)"
    print("\n" + "="*65)
    print("NNPS ORIGINAL PROTOCOL (intentionally inflated)")
    print("Replicating: PCA leakage + random pair split + sum aggregation")
    print(f"Classifier: {classifier}")
    print("="*65 + "\n")

    combo_f, mono, ppi, targets, all_drugs, se_ids = load_decagon_data(se_ids)

    Z_nnps, drug2idx = build_nnps_features_original(
        targets, ppi, mono, all_drugs)

    negatives = build_negatives(combo_f, strategy="random",
                                se_ids=se_ids, neg_ratio=1.0)

    from preprocessing.splitting import Splitter
    splitter = Splitter(combo_f, negatives,
                        strategy="random_pair", seed=RANDOM_SEED)
    splits   = {se: splitter.split(se) for se in se_ids}

    input_dim = Z_nnps.shape[1]  # 525

    if use_lgbm:
        import lightgbm as lgb
        lgbm_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "verbose": -1,
        }

    results_rows = []
    model_label  = "nnps_lgbm" if use_lgbm else "nnps_mlp"
    print(f"\nTraining per-SE {classifier} on {len(se_ids)} side effects...")

    for i, se_id in enumerate(se_ids):
        split = splits[se_id]
        if len(split.train) == 0 or split.train["label"].nunique() < 2:
            continue

        X_tr = make_pair_sum(Z_nnps, drug2idx,
                             split.train["drug_1"].tolist(),
                             split.train["drug_2"].tolist())
        y_tr = split.train["label"].values
        X_vl = make_pair_sum(Z_nnps, drug2idx,
                             split.val["drug_1"].tolist(),
                             split.val["drug_2"].tolist())
        y_vl = split.val["label"].values
        X_te = make_pair_sum(Z_nnps, drug2idx,
                             split.test["drug_1"].tolist(),
                             split.test["drug_2"].tolist())
        y_te = split.test["label"].values

        if use_lgbm:
            n_pos = y_tr.sum(); n_neg = len(y_tr) - n_pos
            params = lgbm_params.copy()
            params["scale_pos_weight"] = n_neg / max(n_pos, 1)
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval   = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)
            booster = lgb.train(
                params, dtrain,
                num_boost_round=500,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(period=-1)],
            )
            proba = booster.predict(X_te)
        else:
            proba = train_mlp_per_se(
                X_tr, y_tr, X_vl, y_vl, X_te, y_te,
                input_dim=input_dim,
            )

        metrics = compute_metrics(y_te, proba)

        results_rows.append({
            "se_id"   : se_id,
            "se_name" : SE_NAMES.get(se_id, se_id),
            "model"   : model_label,
            "protocol": "original (inflated)",
            "split"   : "random_pair",
            "neg"     : "random",
            "operator": "sum",
            "features": "pca_target+mono_525d",
            **metrics,
        })

        if (i + 1) % 3 == 0 or i == 0:
            print(f"  [{i+1:2d}/{len(se_ids)}] {SE_NAMES.get(se_id, se_id):35s} "
                  f"AUROC={metrics['auroc']:.4f}  AP={metrics['ap']:.4f}")

    results = pd.DataFrame(results_rows)
    print(f"\n  Mean AUROC (original protocol): {results['auroc'].mean():.4f}")
    print(f"  Mean AP   (original protocol): {results['ap'].mean():.4f}")

    save_results(results, "nnps_original")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ses", default="rep",
                        help="'rep' for 12 representative SEs, 'all' for 963")
    parser.add_argument("--use-lgbm", action="store_true",
                        help="Use LightGBM instead of the paper's MLP")
    args = parser.parse_args()
    se_ids = None if args.ses == "rep" else "all"
    run_nnps_original(se_ids=se_ids, use_lgbm=args.use_lgbm)
