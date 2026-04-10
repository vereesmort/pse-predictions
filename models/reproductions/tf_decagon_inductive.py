"""
models/reproductions/tf_decagon_inductive.py
=============================================
Reproduction of TF-Decagon (Lloyd et al., 2024) under the FAIR protocol,
converting the transductive model into an inductive one.

Strategy: train DistMult/SimplE on training drugs. For held-out (cold-start)
drugs, learn a linear mapping from drug features (protein targets + FP)
to the embedding space — the same approach used by CSMDDI.

This produces the INDUCTIVE version: predictions can be made for drugs
not seen during training, at the cost of some performance degradation vs
the transductive version. The gap quantifies how much of TF-Decagon's
reported AUROC depends on transductive memorisation.

Usage
-----
    python models/reproductions/tf_decagon_inductive.py
    python models/reproductions/tf_decagon_inductive.py --model simple
    python models/reproductions/tf_decagon_inductive.py --run-both  # compare vs transductive
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import TruncatedSVD

from models.reproductions._utils import (
    load_decagon_data, build_feature_set, build_negatives,
    build_splits, save_results, compare_protocols, SE_NAMES,
)
from models.reproductions.tf_decagon_transductive import (
    DistMult, SimplE, build_triple_index, train_tf_model,
)
from evaluation.metrics import compute_metrics
from configs.config import RANDOM_SEED, CACHE_DIR


def build_drug_feature_matrix(fset, all_drugs, feature_mode="target+fp"):
    """Get drug feature matrix (n_drugs × feat_dim) for linear mapping."""
    return fset.get(all_drugs, mode=feature_mode).astype(np.float32)


def learn_cold_start_mapping(model, drug_feats, train_drug_indices):
    """
    Learn a linear mapping: drug_features → entity_embedding
    using Ridge regression on training drugs.

    For SimplE we map to E1 (head embedding).
    """
    if hasattr(model, "E2"):
        E_train = model.E1[train_drug_indices]
    else:
        E_train = model.E[train_drug_indices]

    X_train = drug_feats[train_drug_indices]
    print(f"  Learning linear feature→embedding mapping: "
          f"{X_train.shape} → {E_train.shape}")

    reg = MultiOutputRegressor(Ridge(alpha=1.0), n_jobs=-1)
    reg.fit(X_train, E_train)
    return reg


def predict_cold_start_embeddings(reg, drug_feats, all_drug_indices,
                                  model, train_drug_indices):
    """
    For cold-start (held-out) drugs: use linear mapping.
    For training drugs: use learned embeddings directly.
    """
    n_drugs = len(all_drug_indices)
    dim = model.E1.shape[1] if hasattr(model, "E2") else model.E.shape[1]
    E_pred = np.zeros((n_drugs, dim), dtype=np.float32)

    train_set = set(train_drug_indices.tolist())
    cold_mask  = np.array([i not in train_set for i in range(n_drugs)])

    # Mapping-predicted embeddings for all (will override train below)
    E_pred = reg.predict(drug_feats).astype(np.float32)

    # Override with learned embeddings for training drugs
    if hasattr(model, "E2"):
        E_pred[train_drug_indices] = model.E1[train_drug_indices]
    else:
        E_pred[train_drug_indices] = model.E[train_drug_indices]

    n_cold = cold_mask.sum()
    print(f"  Cold-start drugs (embedding from linear map): {n_cold} / {n_drugs}")
    return E_pred


def run_tf_decagon_inductive(se_ids=None, model_name="distmult",
                              dim=128, epochs=50,
                              add_self_loops=True,
                              feature_mode="target+fp"):

    protocol = f"tf_decagon_{model_name}_inductive_cold_start"
    print("\n" + "="*65)
    print("TF-DECAGON INDUCTIVE (FAIR) REPRODUCTION")
    print(f"Model: {model_name}  dim={dim}  Feature: {feature_mode}")
    print("Protocol: FAIR — drug cold-start + structured negatives")
    print("Cold-start mapping: linear Ridge from drug features → embeddings")
    print("="*65 + "\n")

    combo_f, mono, ppi, targets, all_drugs, se_ids = load_decagon_data(se_ids)
    all_ses = sorted(combo_f["Polypharmacy Side Effect"].unique())
    d2i     = {d: i for i, d in enumerate(all_drugs)}

    # Drug features for cold-start mapping
    print("Building drug features for cold-start mapping...")
    fset = build_feature_set(targets, ppi, all_drugs)
    drug_feats = build_drug_feature_matrix(fset, all_drugs, feature_mode)

    # Negatives + drug cold-start splits
    negatives = build_negatives(combo_f, "structured", se_ids)
    splits    = build_splits(combo_f, negatives, "drug_cold_start", se_ids)

    # Build triple index (all training drug pairs)
    print("Building triple index...")
    h, r, t, _, s2i, n_rels = build_triple_index(
        combo_f, mono, all_drugs, all_ses, add_self_loops=add_self_loops)

    # Determine training drugs (not in held-out set)
    # Use the cold-start splitter's held-out set
    from preprocessing.splitting import Splitter
    from preprocessing.sampling import NegativeSampler
    splitter_obj = Splitter(combo_f, negatives,
                            strategy="drug_cold_start", seed=RANDOM_SEED)
    held_out_drugs = splitter_obj.held_out_drugs
    train_drugs    = [d for d in all_drugs if d not in held_out_drugs]
    train_d_idx    = np.array([d2i[d] for d in train_drugs], dtype=np.int32)

    # Filter triples to training drugs only
    train_mask = np.isin(h, train_d_idx) & np.isin(t, train_d_idx)
    h_tr, r_tr, t_tr = h[train_mask], r[train_mask], t[train_mask]
    print(f"  Training triples: {len(h_tr):,} (excluding {len(held_out_drugs)} held-out drugs)")

    # Validation: small random subset
    rng   = np.random.default_rng(RANDOM_SEED)
    n_val = min(5000, len(h_tr) // 10)
    vi    = rng.choice(len(h_tr), n_val, replace=False)
    h_val, r_val, t_val = h_tr[vi], r_tr[vi], t_tr[vi]
    y_val = np.ones(len(h_val))

    # Initialise and train model (training drugs only)
    MODEL_CLS = {"distmult": DistMult, "simple": SimplE}
    model = MODEL_CLS[model_name](len(all_drugs), n_rels, dim=dim)
    print(f"\nTraining {model_name} on training drugs ({len(train_drugs)} drugs)...")
    model = train_tf_model(model, h_tr, r_tr, t_tr,
                           h_val, r_val, t_val, y_val,
                           epochs=epochs, batch_size=1024)

    # Learn cold-start mapping
    print("\nLearning cold-start mapping (feature → embedding)...")
    cs_map = learn_cold_start_mapping(model, drug_feats, train_d_idx)
    E_inductive = predict_cold_start_embeddings(
        cs_map, drug_feats, list(range(len(all_drugs))), model, train_d_idx)

    # Get relation embeddings
    R_mat = model.R1 if hasattr(model, "R2") else model.R

    def score_inductive(h_idx, r_idx, t_idx):
        if hasattr(model, "R2"):
            s1 = (E_inductive[h_idx] * model.R1[r_idx] * E_inductive[t_idx]).sum(axis=-1)
            s2 = (E_inductive[h_idx] * model.R2[r_idx] * E_inductive[t_idx]).sum(axis=-1)
            return 0.5 * (s1 + s2)
        return (E_inductive[h_idx] * R_mat[r_idx] * E_inductive[t_idx]).sum(axis=-1)

    # Evaluate per SE on test pairs from cold-start splits
    print("\nEvaluating per SE (cold-start test set)...")
    results_rows = []
    for se_id in se_ids:
        if se_id not in s2i:
            continue
        split = splits[se_id]
        if len(split.test) == 0:
            continue
        ri = s2i[se_id]

        test = split.test
        h_idx = np.array([d2i.get(d, 0) for d in test["drug_1"]], dtype=np.int32)
        t_idx = np.array([d2i.get(d, 0) for d in test["drug_2"]], dtype=np.int32)
        r_idx = np.full(len(h_idx), ri, dtype=np.int32)
        y_true = test["label"].values

        scores = score_inductive(h_idx, r_idx, t_idx)
        proba  = expit(scores)
        if len(np.unique(y_true)) < 2:
            continue

        metrics = compute_metrics(y_true, proba)
        results_rows.append({
            "se_id"   : se_id,
            "se_name" : SE_NAMES.get(se_id, se_id),
            "model"   : f"{model_name}_inductive",
            "protocol": protocol,
            "split"   : "drug_cold_start",
            "neg"     : "structured",
            "self_loops": add_self_loops,
            "feature_mode": feature_mode,
            **metrics,
        })

    results = pd.DataFrame(results_rows)
    if len(results):
        print(f"\n  Mean AUROC ({protocol}): {results['auroc'].mean():.4f}")
        print(f"  Mean AP   ({protocol}): {results['ap'].mean():.4f}")
    save_results(results, f"tf_decagon_{model_name}_inductive")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ses",          default="rep")
    parser.add_argument("--model",        default="distmult",
                        choices=["distmult", "simple"])
    parser.add_argument("--dim",          type=int, default=128)
    parser.add_argument("--epochs",       type=int, default=50)
    parser.add_argument("--feature-mode", default="target+fp")
    parser.add_argument("--no-selfloops", action="store_true")
    parser.add_argument("--run-both",     action="store_true",
                        help="Run transductive + inductive and compare")
    args = parser.parse_args()
    se_ids = None if args.ses == "rep" else "all"

    if args.run_both:
        from models.reproductions.tf_decagon_transductive import \
            run_tf_decagon_transductive
        r_trans = run_tf_decagon_transductive(
            se_ids, args.model, args.dim, args.epochs,
            add_self_loops=not args.no_selfloops)
        r_ind   = run_tf_decagon_inductive(
            se_ids, args.model, args.dim, args.epochs,
            add_self_loops=not args.no_selfloops,
            feature_mode=args.feature_mode)
        r_trans["protocol"] = "original (transductive, inflated)"
        r_ind["protocol"]   = "fair (inductive, cold-start)"
        combined = pd.concat([r_trans, r_ind], ignore_index=True)
        compare_protocols(combined, f"TF-Decagon ({args.model})")
        save_results(combined, f"tf_decagon_{args.model}_combined")
    else:
        run_tf_decagon_inductive(
            se_ids, args.model, args.dim, args.epochs,
            add_self_loops=not args.no_selfloops,
            feature_mode=args.feature_mode)
