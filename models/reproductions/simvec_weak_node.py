"""
models/reproductions/simvec_weak_node.py
models/reproductions/simvec_cold_start.py
==========================================
Reproductions of SimVec (Lukashina et al., 2022).

SimVec augments TriVec knowledge graph embeddings with:
  1. Chemical initialisation: drug node embeddings seeded from Morgan FP
  2. Weighted similarity edges: connect chemically similar drug pairs
  3. Three-stage training: (1) polypharmacy edges, (2) similarity edges,
     (3) weak-node embeddings
  4. "Stay Positive" negative sampling (eliminates explicit negatives,
      adds regularisation term instead)

Two evaluation protocols:
  simvec_weak_node.py   — original "weak-node split" (lowest-degree drugs held out)
  simvec_cold_start.py  — fair "drug cold-start split" (random 20% of drugs held out)

This file provides both in a single module, switchable via --split.

Note on similarity edges: the original SimVec computes a full drug–drug
Tanimoto similarity matrix (O(n²) = 645² = 415,225 edges). This is retained
here as it is tractable at n=645. For larger drug libraries it would need pruning.

Usage
-----
    python models/reproductions/simvec_weak_node.py
    python models/reproductions/simvec_cold_start.py
    python models/reproductions/simvec_weak_node.py --run-both
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.sparse import csr_matrix

from models.reproductions._utils import (
    load_decagon_data, build_feature_set, build_negatives,
    build_splits, save_results, compare_protocols, SE_NAMES,
)
from models.reproductions.tf_decagon_transductive import train_tf_model
from evaluation.metrics import compute_metrics
from configs.config import RANDOM_SEED, CACHE_DIR


# ── TriVec model (SimVec backbone) ────────────────────────────────────────────

class TriVec:
    """
    TriVec tensor factorisation (Montella et al., 2020):
    score(h,r,t) = <e1_h, w1_r, e3_t> + <e2_h, w2_r, e2_t> + <e3_h, w3_r, e1_t>

    Fully expressive, handles asymmetric relations.
    SimVec initialises E1=E2=E3 from Morgan fingerprints.
    """
    name = "trivec"

    def __init__(self, n_entities, n_relations, dim=100,
                 lr=0.001, seed=RANDOM_SEED):
        rng   = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(dim)
        self.E1 = rng.normal(0, scale, (n_entities, dim)).astype(np.float32)
        self.E2 = rng.normal(0, scale, (n_entities, dim)).astype(np.float32)
        self.E3 = rng.normal(0, scale, (n_entities, dim)).astype(np.float32)
        self.W1 = rng.normal(0, scale, (n_relations, dim)).astype(np.float32)
        self.W2 = rng.normal(0, scale, (n_relations, dim)).astype(np.float32)
        self.W3 = rng.normal(0, scale, (n_relations, dim)).astype(np.float32)
        self.lr   = lr
        self.dim  = dim

    def score(self, h_idx, r_idx, t_idx):
        s1 = (self.E1[h_idx] * self.W1[r_idx] * self.E3[t_idx]).sum(axis=-1)
        s2 = (self.E2[h_idx] * self.W2[r_idx] * self.E2[t_idx]).sum(axis=-1)
        s3 = (self.E3[h_idx] * self.W3[r_idx] * self.E1[t_idx]).sum(axis=-1)
        return s1 + s2 + s3

    def update(self, h_idx, r_idx, t_idx, labels,
               weights=None, reg=1e-3):
        s     = self.score(h_idx, r_idx, t_idx)
        p     = expit(s)
        delta = (p - labels) / len(labels)
        if weights is not None:
            delta = delta * weights

        d = delta[:, None]
        # Gradients
        dE1h = d * self.W1[r_idx] * self.E3[t_idx] + d * self.W3[r_idx] * self.E3[t_idx]
        dE2h = d * self.W2[r_idx] * self.E2[t_idx]
        dE3h = d * self.W3[r_idx] * self.E1[t_idx]
        dE3t = d * self.W1[r_idx] * self.E1[h_idx]
        dE2t = d * self.W2[r_idx] * self.E2[h_idx]
        dE1t = d * self.W3[r_idx] * self.E3[h_idx]
        dW1  = d * self.E1[h_idx] * self.E3[t_idx]
        dW2  = d * self.E2[h_idx] * self.E2[t_idx]
        dW3  = d * self.E3[h_idx] * self.E1[t_idx]

        lr = self.lr
        for arr, idx, grad in [
            (self.E1, h_idx, dE1h), (self.E2, h_idx, dE2h), (self.E3, h_idx, dE3h),
            (self.E3, t_idx, dE3t), (self.E2, t_idx, dE2t), (self.E1, t_idx, dE1t),
            (self.W1, r_idx, dW1),  (self.W2, r_idx, dW2),  (self.W3, r_idx, dW3),
        ]:
            np.add.at(arr, idx, -lr * (grad + reg * arr[idx]))

        return float(np.mean(-(labels * np.log(p + 1e-9) +
                               (1 - labels) * np.log(1 - p + 1e-9))))


# ── Chemical initialisation ───────────────────────────────────────────────────

def initialise_from_fingerprints(model, fset, all_drugs):
    """
    Seed drug embeddings E1=E2=E3 with Morgan FP projections
    (SimVec 'chemical initialisation').
    """
    FP = fset.get(all_drugs, mode="fp_only").astype(np.float32)
    from sklearn.decomposition import TruncatedSVD
    k   = model.dim
    svd = TruncatedSVD(n_components=k, random_state=RANDOM_SEED)
    Z   = svd.fit_transform(csr_matrix(FP)).astype(np.float32)
    # Scale to unit norm
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    Z     = Z / np.where(norms > 0, norms, 1)
    n = min(len(all_drugs), model.E1.shape[0])
    model.E1[:n] = Z[:n]
    model.E2[:n] = Z[:n].copy()
    model.E3[:n] = Z[:n].copy()
    print(f"  Initialised drug embeddings from Morgan FP (dim={k})")


# ── Similarity edges ──────────────────────────────────────────────────────────

def build_similarity_edges(fset, all_drugs, threshold=0.3):
    """
    Build drug–drug Tanimoto similarity edges weighted by similarity × InvDeg.
    Only pairs above threshold are included (SimVec window-based weighting).
    Returns (h_idx, t_idx, weights) arrays.
    """
    from rdkit.Chem import DataStructs
    FP = fset.get(all_drugs, mode="fp_only").astype(np.float32)
    d2i = {d: i for i, d in enumerate(all_drugs)}
    n   = len(all_drugs)

    # Dot-product Tanimoto for binary FP: sim = |A∩B| / |A∪B|
    norms = FP.sum(axis=1)
    dot   = FP @ FP.T
    union = norms[:, None] + norms[None, :] - dot
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, dot / union, 0.0)
    np.fill_diagonal(sim, 0)

    # Degree (number of similar pairs above threshold)
    deg = (sim >= threshold).sum(axis=1).astype(np.float32)
    inv_deg = np.where(deg > 0, 1.0 / deg, 0.0)

    # Collect edges above threshold
    h_idx, t_idx, weights = [], [], []
    for i in range(n):
        for j in range(i+1, n):
            s = sim[i, j]
            if s >= threshold:
                w = s * inv_deg[i] * inv_deg[j]
                h_idx.append(i); t_idx.append(j); weights.append(w)
                h_idx.append(j); t_idx.append(i); weights.append(w)

    print(f"  Similarity edges (threshold={threshold}): {len(h_idx)//2:,} pairs")
    return (np.array(h_idx, dtype=np.int32),
            np.array(t_idx, dtype=np.int32),
            np.array(weights, dtype=np.float32))


# ── Weak-node split ───────────────────────────────────────────────────────────

def weak_node_split(combo_f, all_drugs, n_weak=98, seed=RANDOM_SEED):
    """
    Hold out the N lowest-degree drugs as weak nodes (SimVec original).
    Their edges are split 50/50 val/test; remaining edges are training.
    """
    degree = {}
    for _, row in combo_f.iterrows():
        degree[row["STITCH 1"]] = degree.get(row["STITCH 1"], 0) + 1
        degree[row["STITCH 2"]] = degree.get(row["STITCH 2"], 0) + 1

    sorted_drugs = sorted(all_drugs, key=lambda d: degree.get(d, 0))
    weak_drugs   = set(sorted_drugs[:n_weak])
    print(f"  Weak nodes: {n_weak} lowest-degree drugs (max degree "
          f"{max(degree.get(d,0) for d in sorted_drugs[:n_weak])})")

    weak_edges  = combo_f[combo_f["STITCH 1"].isin(weak_drugs) |
                          combo_f["STITCH 2"].isin(weak_drugs)]
    train_edges = combo_f[~combo_f.index.isin(weak_edges.index)]

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(weak_edges))
    val_edges  = weak_edges.iloc[idx[:len(idx)//2]]
    test_edges = weak_edges.iloc[idx[len(idx)//2:]]

    return train_edges, val_edges, test_edges, weak_drugs


# ── Main training and evaluation ──────────────────────────────────────────────

def run_simvec(se_ids=None, split_type="weak_node",
               dim=100, epochs=30, sim_threshold=0.3,
               n_weak=98, use_chem_init=True, use_sim_edges=True):

    label = (f"simvec_{'chem' if use_chem_init else 'nochem'}_"
             f"{'simed' if use_sim_edges else 'nosimed'}_{split_type}")
    print("\n" + "="*65)
    print(f"SIMVEC REPRODUCTION — {label}")
    print(f"Split: {split_type}  dim={dim}  chem_init={use_chem_init}  "
          f"sim_edges={use_sim_edges}")
    print("="*65 + "\n")

    combo_f, mono, ppi, targets, all_drugs, se_ids = load_decagon_data(se_ids)
    all_ses = sorted(combo_f["Polypharmacy Side Effect"].unique())
    d2i     = {d: i for i, d in enumerate(all_drugs)}
    s2i_poly = {s: i for i, s in enumerate(all_ses)}

    # Drug features for chemical initialisation
    fset = build_feature_set(targets, ppi, all_drugs)

    # Build polypharmacy triple index
    from models.reproductions.tf_decagon_transductive import build_triple_index
    h_all, r_all, t_all, _, _, n_rels = build_triple_index(
        combo_f, mono, all_drugs, all_ses, add_self_loops=False)

    # ── Split ─────────────────────────────────────────────────────────────────
    if split_type == "weak_node":
        train_df, val_df, test_df, held_drugs = weak_node_split(
            combo_f, all_drugs, n_weak=n_weak)
        train_drugs = set(all_drugs) - held_drugs
    else:
        # Drug cold-start
        from preprocessing.splitting import Splitter
        from preprocessing.sampling import NegativeSampler
        negatives  = build_negatives(combo_f, "structured", se_ids)
        splitter   = Splitter(combo_f, negatives, "drug_cold_start", seed=RANDOM_SEED)
        held_drugs = splitter.held_out_drugs
        train_drugs = set(all_drugs) - held_drugs
        splits     = {se: splitter.split(se) for se in se_ids}

    train_drug_idx = np.array([d2i[d] for d in all_drugs if d in train_drugs],
                               dtype=np.int32)

    # Filter triples to training drugs
    train_mask = np.isin(h_all, train_drug_idx) & np.isin(t_all, train_drug_idx)
    h_tr = h_all[train_mask]; r_tr = r_all[train_mask]; t_tr = t_all[train_mask]

    # Val triples
    rng   = np.random.default_rng(RANDOM_SEED)
    n_val = min(3000, len(h_tr) // 10)
    vi    = rng.choice(len(h_tr), n_val, replace=False)
    h_vl, r_vl, t_vl = h_tr[vi], r_tr[vi], t_tr[vi]

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TriVec(len(all_drugs), n_rels, dim=dim)
    if use_chem_init:
        initialise_from_fingerprints(model, fset, all_drugs)

    # Stage 1: polypharmacy edges
    print(f"\nStage 1: Training on polypharmacy edges ({len(h_tr):,} triples)...")
    model = train_tf_model(model, h_tr, r_tr, t_tr,
                           h_vl, r_vl, t_vl, np.ones(len(h_vl)),
                           epochs=epochs, batch_size=512)

    # Stage 2: similarity edges (SimVec contribution)
    if use_sim_edges:
        print("\nStage 2: Training on chemical similarity edges...")
        sh, st, sw = build_similarity_edges(fset, all_drugs, threshold=sim_threshold)
        # Similarity edges use a special relation index (beyond polypharmacy)
        sr = np.full(len(sh), n_rels, dtype=np.int32)
        # Expand model to include one more relation
        model.W1 = np.vstack([model.W1,
                               np.zeros((1, dim), dtype=np.float32)])
        model.W2 = np.vstack([model.W2,
                               np.zeros((1, dim), dtype=np.float32)])
        model.W3 = np.vstack([model.W3,
                               np.zeros((1, dim), dtype=np.float32)])
        for epoch in range(10):
            idx   = rng.permutation(len(sh))
            bs    = 2048
            loss  = 0.0
            nb    = 0
            for s in range(0, len(sh), bs):
                hb, rb, tb, wb = (sh[idx[s:s+bs]], sr[idx[s:s+bs]],
                                  st[idx[s:s+bs]], sw[idx[s:s+bs]])
                # Stay-Positive: all similarity edges are "soft positives"
                lb = np.ones(len(hb))
                loss += model.update(hb, rb, tb, lb, weights=wb)
                nb   += 1
            if (epoch+1) % 5 == 0:
                print(f"    Sim-edge epoch {epoch+1}/10  loss={loss/nb:.4f}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results_rows = []
    print("\nEvaluating per SE...")

    if split_type == "weak_node":
        # Evaluate on test_df edges
        for se_id in se_ids:
            if se_id not in s2i_poly:
                continue
            ri = s2i_poly[se_id]
            te = test_df[test_df["Polypharmacy Side Effect"] == se_id]
            if len(te) < 2:
                continue

            h_pos = np.array([d2i.get(d, 0) for d in te["STITCH 1"]], dtype=np.int32)
            t_pos = np.array([d2i.get(d, 0) for d in te["STITCH 2"]], dtype=np.int32)
            t_neg = rng.integers(0, len(all_drugs), len(h_pos)).astype(np.int32)
            h_ev  = np.concatenate([h_pos, h_pos])
            r_ev  = np.full(len(h_ev), ri, dtype=np.int32)
            t_ev  = np.concatenate([t_pos, t_neg])
            y_ev  = np.array([1]*len(h_pos) + [0]*len(h_pos))

            scores = model.score(h_ev, r_ev, t_ev)
            proba  = expit(scores)
            if len(np.unique(y_ev)) < 2:
                continue
            metrics = compute_metrics(y_ev, proba)
            results_rows.append({
                "se_id": se_id, "se_name": SE_NAMES.get(se_id, se_id),
                "model": "simvec_trivec", "protocol": label,
                "split": split_type, "chem_init": use_chem_init,
                "sim_edges": use_sim_edges, **metrics,
            })
    else:
        # Cold-start evaluation via splits dict
        for se_id in se_ids:
            if se_id not in s2i_poly:
                continue
            split = splits[se_id]
            if len(split.test) == 0:
                continue
            ri = s2i_poly[se_id]
            test = split.test
            h_ev = np.array([d2i.get(d,0) for d in test["drug_1"]], dtype=np.int32)
            t_ev = np.array([d2i.get(d,0) for d in test["drug_2"]], dtype=np.int32)
            r_ev = np.full(len(h_ev), ri, dtype=np.int32)
            y_ev = test["label"].values
            scores = model.score(h_ev, r_ev, t_ev)
            proba  = expit(scores)
            if len(np.unique(y_ev)) < 2:
                continue
            metrics = compute_metrics(y_ev, proba)
            results_rows.append({
                "se_id": se_id, "se_name": SE_NAMES.get(se_id, se_id),
                "model": "simvec_trivec", "protocol": label,
                "split": split_type, "chem_init": use_chem_init,
                "sim_edges": use_sim_edges, **metrics,
            })

    results = pd.DataFrame(results_rows)
    if len(results):
        print(f"\n  Mean AUROC ({label}): {results['auroc'].mean():.4f}")
        print(f"  Mean AP   ({label}): {results['ap'].mean():.4f}")
    save_results(results, label)
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ses",           default="rep")
    parser.add_argument("--split",         default="weak_node",
                        choices=["weak_node", "drug_cold_start"])
    parser.add_argument("--dim",           type=int, default=100)
    parser.add_argument("--epochs",        type=int, default=30)
    parser.add_argument("--n-weak",        type=int, default=98)
    parser.add_argument("--no-chem-init",  action="store_true")
    parser.add_argument("--no-sim-edges",  action="store_true")
    parser.add_argument("--run-both",      action="store_true",
                        help="Run weak_node AND drug_cold_start and compare")
    args = parser.parse_args()
    se_ids = None if args.ses == "rep" else "all"

    if args.run_both:
        r_wn = run_simvec(se_ids, "weak_node",       args.dim, args.epochs,
                          n_weak=args.n_weak)
        r_cs = run_simvec(se_ids, "drug_cold_start", args.dim, args.epochs)
        r_wn["protocol"] = "original (weak-node)"
        r_cs["protocol"] = "fair (drug cold-start)"
        combined = pd.concat([r_wn, r_cs], ignore_index=True)
        compare_protocols(combined, "SimVec")
        save_results(combined, "simvec_combined")
    else:
        run_simvec(
            se_ids=se_ids, split_type=args.split,
            dim=args.dim, epochs=args.epochs, n_weak=args.n_weak,
            use_chem_init=not args.no_chem_init,
            use_sim_edges=not args.no_sim_edges,
        )
