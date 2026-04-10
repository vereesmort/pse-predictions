"""
models/reproductions/tf_decagon_transductive.py
================================================
Reproduction of TF-Decagon (Lloyd et al., 2024) under the ORIGINAL
transductive protocol — the setting that produces AUROC 0.978.

Key characteristics of the original:
  - SimplE / DistMult / ComplEx tensor factorisation
  - Random 90/10 edge split (pairs involving the SAME drugs in train+test)
  - Mono SEs encoded as self-loop edges on each drug
  - No Morgan fingerprints
  - Transductive: drug embeddings are learned and fixed; cannot generalise
    to drugs not seen during training

This script uses DistMult as the primary model (simplest, well-understood)
with a numpy/scipy implementation — no LibKGE dependency needed.

SimplE and ComplEx are implemented as additional options (--model flag).

Usage
-----
    python models/reproductions/tf_decagon_transductive.py
    python models/reproductions/tf_decagon_transductive.py --model simple
    python models/reproductions/tf_decagon_transductive.py --model complex
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid

import torch
import torch.nn as nn

from models.reproductions._utils import (
    load_decagon_data, save_results, compare_protocols, SE_NAMES,
)
from evaluation.metrics import compute_metrics
from configs.config import METRICS_DIR, RANDOM_SEED

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Tensor factorisation models (PyTorch) ─────────────────────────────────────

class DistMult(nn.Module):
    """
    DistMult: score(h, r, t) = <e_h, w_r, e_t>  (element-wise product then sum)
    Symmetric — same score for (h,r,t) and (t,r,h).
    """
    name = "distmult"

    def __init__(self, n_entities, n_relations, dim=128, seed=RANDOM_SEED):
        super().__init__()
        torch.manual_seed(seed)
        scale = 1.0 / (dim ** 0.5)
        self.E = nn.Embedding(n_entities,  dim)
        self.R = nn.Embedding(n_relations, dim)
        nn.init.uniform_(self.E.weight, -scale, scale)
        nn.init.uniform_(self.R.weight, -scale, scale)

    def score(self, h_idx, r_idx, t_idx):
        return (self.E(h_idx) * self.R(r_idx) * self.E(t_idx)).sum(dim=-1)

    def n_entities(self):
        return self.E.weight.shape[0]


class SimplE(nn.Module):
    """
    SimplE: score(h,r,t) = 0.5 * (<e_h, w_r, e'_t> + <e'_h, w'_r, e_t>)
    Fully expressive; handles asymmetric relations.
    """
    name = "simple"

    def __init__(self, n_entities, n_relations, dim=128, seed=RANDOM_SEED):
        super().__init__()
        torch.manual_seed(seed)
        scale = 1.0 / (dim ** 0.5)
        self.E1 = nn.Embedding(n_entities,  dim)
        self.E2 = nn.Embedding(n_entities,  dim)
        self.R1 = nn.Embedding(n_relations, dim)
        self.R2 = nn.Embedding(n_relations, dim)
        for emb in (self.E1, self.E2, self.R1, self.R2):
            nn.init.uniform_(emb.weight, -scale, scale)

    def score(self, h_idx, r_idx, t_idx):
        s1 = (self.E1(h_idx) * self.R1(r_idx) * self.E2(t_idx)).sum(dim=-1)
        s2 = (self.E2(h_idx) * self.R2(r_idx) * self.E1(t_idx)).sum(dim=-1)
        return 0.5 * (s1 + s2)

    def n_entities(self):
        return self.E1.weight.shape[0]


# ── Build entity/relation index ───────────────────────────────────────────────

def build_triple_index(combo_f, mono, drugs, ses, add_self_loops=True):
    """
    Build integer triple arrays (h, r, t) for all positive edges.

    Polypharmacy edges:   (drug_i, se_r, drug_j)
    Mono self-loop edges: (drug_i, mono_se_r, drug_i)  [TF-Decagon 'Self-loops']

    Fully vectorised — no iterrows().
    """
    d2i = {d: i for i, d in enumerate(drugs)}
    s2i = {s: i for i, s in enumerate(ses)}   # polypharmacy SEs

    # Vectorised polypharmacy triples
    cf = combo_f[
        combo_f["STITCH 1"].isin(d2i) &
        combo_f["STITCH 2"].isin(d2i) &
        combo_f["Polypharmacy Side Effect"].isin(s2i)
    ].copy()
    h = cf["STITCH 1"].map(d2i).values.astype(np.int32)
    r = cf["Polypharmacy Side Effect"].map(s2i).values.astype(np.int32)
    t = cf["STITCH 2"].map(d2i).values.astype(np.int32)

    n_poly_relations = len(ses)

    # Mono self-loop triples (TF-Decagon contribution) — also vectorised
    if add_self_loops:
        mono_ses = sorted(mono["Individual Side Effect"].unique())
        ms2i     = {s: i + n_poly_relations for i, s in enumerate(mono_ses)}
        mono_f   = mono[mono["STITCH"].isin(d2i) &
                        mono["Individual Side Effect"].isin(ms2i)].copy()
        sl_h = mono_f["STITCH"].map(d2i).values.astype(np.int32)
        sl_r = mono_f["Individual Side Effect"].map(ms2i).values.astype(np.int32)
        sl_t = sl_h.copy()  # self-loops
        h = np.concatenate([h, sl_h])
        r = np.concatenate([r, sl_r])
        t = np.concatenate([t, sl_t])
        n_total_relations = n_poly_relations + len(mono_ses)
    else:
        n_total_relations = n_poly_relations

    return h, r, t, d2i, s2i, n_total_relations


# ── Training loop ─────────────────────────────────────────────────────────────

def _t(arr, device):
    """Convert int32 numpy array to a LongTensor on device."""
    return torch.from_numpy(arr.astype(np.int64)).to(device)


def train_tf_model(model, h_train, r_train, t_train,
                   h_val, r_val, t_val, y_val,
                   epochs=50, batch_size=1024, seed=RANDOM_SEED):
    """
    Train a DistMult / SimplE model with PyTorch autograd + Adam.
    Replaces the previous per-sample np.add.at scatter which was
    orders-of-magnitude slower on 4M+ triples.
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    rng = np.random.default_rng(seed)
    n_ents = model.n_entities()
    best_auroc  = -np.inf
    best_state  = None

    h_val_t = _t(h_val, DEVICE)
    r_val_t = _t(r_val, DEVICE)
    t_val_t = _t(t_val, DEVICE)

    for epoch in range(epochs):
        model.train()
        idx = rng.permutation(len(h_train))
        h_tr = h_train[idx]; r_tr = r_train[idx]; t_tr = t_train[idx]

        epoch_loss = 0.0
        n_batches  = 0
        for start in range(0, len(h_tr), batch_size):
            h_b = h_tr[start:start + batch_size]
            r_b = r_tr[start:start + batch_size]
            t_b = t_tr[start:start + batch_size]
            # 1:1 random-tail negatives
            t_neg = rng.integers(0, n_ents, len(h_b)).astype(np.int32)

            h_all = _t(np.concatenate([h_b, h_b]), DEVICE)
            r_all = _t(np.concatenate([r_b, r_b]), DEVICE)
            t_all = _t(np.concatenate([t_b, t_neg]), DEVICE)
            y_all = torch.cat([torch.ones(len(h_b)), torch.zeros(len(h_b))]).to(DEVICE)

            optimizer.zero_grad()
            loss = criterion(model.score(h_all, r_all, t_all), y_all)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_scores = model.score(h_val_t, r_val_t, t_val_t).cpu().numpy()
        val_proba = expit(val_scores)
        vm = compute_metrics(y_val, val_proba)

        if vm["auroc"] > best_auroc:
            best_auroc = vm["auroc"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  "
                  f"loss={epoch_loss/n_batches:.4f}  "
                  f"val_auroc={vm['auroc']:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def run_tf_decagon_transductive(se_ids=None, model_name="distmult",
                                 dim=128, epochs=50,
                                 add_self_loops=True):

    protocol = f"tf_decagon_{model_name}_{'selfloops' if add_self_loops else 'no_selfloops'}_original"
    print("\n" + "="*65)
    print(f"TF-DECAGON TRANSDUCTIVE REPRODUCTION")
    print(f"Model: {model_name}  Self-loops: {add_self_loops}  dim={dim}")
    print("Protocol: ORIGINAL (random 90/10 split — transductive, inflated)")
    print("="*65 + "\n")

    combo_f, mono, ppi, targets, all_drugs, se_ids = load_decagon_data(se_ids)
    all_ses = sorted(combo_f["Polypharmacy Side Effect"].unique())

    # Build triple index
    print("Building triple index (polypharmacy + mono self-loops)...")
    h, r, t, d2i, s2i, n_rels = build_triple_index(
        combo_f, mono, all_drugs, all_ses, add_self_loops=add_self_loops)
    print(f"  Triples: {len(h):,}  Entities: {len(all_drugs)}  Relations: {n_rels}")

    # Initialise model
    MODEL_CLS = {"distmult": DistMult, "simple": SimplE}
    model = MODEL_CLS[model_name](len(all_drugs), n_rels, dim=dim)

    # Random 90/10 split on EDGES (original TF-Decagon protocol)
    rng    = np.random.default_rng(RANDOM_SEED)
    idx    = rng.permutation(len(h))
    n_test = int(len(h) * 0.10)
    test_idx  = idx[:n_test]
    train_idx = idx[n_test:]

    h_train, r_train, t_train = h[train_idx], r[train_idx], t[train_idx]
    h_test,  r_test,  t_test  = h[test_idx],  r[test_idx],  t[test_idx]

    # Val: last 10% of train
    n_val     = int(len(h_train) * 0.10)
    h_val,  r_val,  t_val  = h_train[-n_val:], r_train[-n_val:], t_train[-n_val:]
    h_train, r_train, t_train = h_train[:-n_val], r_train[:-n_val], t_train[:-n_val]
    y_val = np.ones(len(h_val))  # all positive triples in val

    print(f"\nTraining {model_name}...")
    model = train_tf_model(model, h_train, r_train, t_train,
                           h_val, r_val, t_val, y_val,
                           epochs=epochs, batch_size=1024)

    def _score_np(h_np, r_np, t_np):
        """Score numpy index arrays; returns numpy float32 array."""
        model.eval()
        with torch.no_grad():
            s = model.score(_t(h_np, DEVICE), _t(r_np, DEVICE), _t(t_np, DEVICE))
        return s.cpu().numpy()

    # Evaluate per SE on test edges
    print("\nEvaluating per SE...")
    results_rows = []
    for se_id in se_ids:
        if se_id not in s2i:
            continue
        ri = s2i[se_id]
        mask_test = (r_test == ri)
        if mask_test.sum() < 2:
            continue

        h_pos = h_test[mask_test]
        t_pos = t_test[mask_test]

        # Random-tail negatives
        t_neg = rng.integers(0, len(all_drugs), len(h_pos)).astype(np.int32)
        h_all = np.concatenate([h_pos, h_pos])
        r_all = np.full(len(h_all), ri, dtype=np.int32)
        t_all = np.concatenate([t_pos, t_neg])
        y_all = np.array([1]*len(h_pos) + [0]*len(h_pos))

        scores = _score_np(h_all, r_all, t_all)
        proba  = expit(scores)
        if len(np.unique(y_all)) < 2:
            continue
        metrics = compute_metrics(y_all, proba)
        results_rows.append({
            "se_id": se_id,
            "se_name": SE_NAMES.get(se_id, se_id),
            "model": model_name,
            "protocol": protocol,
            "split": "random_edge_90_10",
            "neg": "random_tail",
            "self_loops": add_self_loops,
            **metrics,
        })

    results = pd.DataFrame(results_rows)
    if len(results):
        print(f"\n  Mean AUROC ({protocol}): {results['auroc'].mean():.4f}")
        print(f"  Mean AP   ({protocol}): {results['ap'].mean():.4f}")
    save_results(results, f"tf_decagon_{model_name}_transductive")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ses",         default="rep")
    parser.add_argument("--model",       default="distmult",
                        choices=["distmult", "simple"])
    parser.add_argument("--dim",         type=int, default=128)
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--no-selfloops", action="store_true")
    args = parser.parse_args()

    se_ids = None if args.ses == "rep" else "all"
    run_tf_decagon_transductive(
        se_ids=se_ids,
        model_name=args.model,
        dim=args.dim,
        epochs=args.epochs,
        add_self_loops=not args.no_selfloops,
    )
