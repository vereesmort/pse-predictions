"""
models/rgcn_model.py
====================
Relational Graph Convolutional Network (RGCN) for polypharmacy side effect
prediction. Implements basis decomposition for scalability across 963 relations,
with mini-batch training (GraphSAINT-style edge sampling).

Architecture
------------
  Input node features (drugs + proteins)
  → L × RGCN layers (relation-aware message passing)
  → Drug embeddings
  → Bilinear decoder per side effect (W_r = sum_b a_{rb} V_b)
  → Sigmoid probability

Design choices vs. Decagon
---------------------------
  - Basis decomposition: W_r = Σ_b a_{rb} V_b  →  O(B·D²) not O(R·D²)
  - Mini-batch edge sampling: avoids full-graph GPU memory
  - PyTorch native (no TF1 dependency)
  - Trains in hours not days on a single GPU / CPU

Requirements: torch (pre-installed)

Usage
-----
    from models.rgcn_model import RGCNPredictor
    model = RGCNPredictor(feature_set, drug_list, se_list)
    model.train(edge_index, edge_type, train_triples, val_triples)
    results = model.evaluate_all(splits, se_ids)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd

from configs.config import RGCN_PARAMS, RANDOM_SEED
from evaluation.metrics import compute_metrics

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[rgcn_model] PyTorch not available — RGCNPredictor will raise on init.")


# ── RGCN Layer ─────────────────────────────────────────────────────────────────

class RGCNLayer(nn.Module):
    """
    Single RGCN layer with basis decomposition.

    For each relation r: W_r = sum_b a_{r,b} * V_b
    Message passing: h_i' = σ( sum_r sum_{j in N_r(i)} W_r h_j / |N_r(i)| + W_0 h_i )
    """

    def __init__(self, in_dim: int, out_dim: int,
                 n_relations: int, n_bases: int, dropout: float = 0.3):
        super().__init__()
        self.in_dim      = in_dim
        self.out_dim     = out_dim
        self.n_relations = n_relations
        self.n_bases     = n_bases

        # Basis matrices: (n_bases, in_dim, out_dim)
        self.V = nn.Parameter(torch.FloatTensor(n_bases, in_dim, out_dim))
        # Basis coefficients: (n_relations, n_bases)
        self.A = nn.Parameter(torch.FloatTensor(n_relations, n_bases))
        # Self-connection weight
        self.W0 = nn.Linear(in_dim, out_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.V.data.view(self.n_bases, -1)
                                .unsqueeze(0)).squeeze()
        nn.init.xavier_uniform_(self.A.unsqueeze(0)).squeeze()

    def forward(self, h: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        """
        h          : (n_nodes, in_dim)
        edge_index : (2, n_edges) — [src, dst]
        edge_type  : (n_edges,)   — relation index per edge

        Returns (n_nodes, out_dim)
        """
        n_nodes = h.size(0)
        out     = torch.zeros(n_nodes, self.out_dim, device=h.device)

        # Relation-wise message passing
        for r in range(self.n_relations):
            mask   = (edge_type == r)
            if mask.sum() == 0:
                continue
            src    = edge_index[0, mask]
            dst    = edge_index[1, mask]
            # Compute W_r from bases
            W_r    = torch.einsum("b,bdi->di", self.A[r], self.V)  # (in, out)
            # Messages: h_src @ W_r
            msg    = self.dropout(h[src]) @ W_r                    # (n_r, out)
            # Aggregate (mean) at destination nodes
            out.index_add_(0, dst, msg)
            # Normalise by degree
            deg    = torch.zeros(n_nodes, device=h.device)
            deg.index_add_(0, dst, torch.ones(mask.sum(), device=h.device))
            deg    = deg.clamp(min=1).unsqueeze(1)
            out    = out / deg

        # Self-connection
        out = out + self.W0(h)
        return F.relu(out)


# ── RGCN Model ─────────────────────────────────────────────────────────────────

class RGCN(nn.Module):
    """
    Multi-layer RGCN encoder + bilinear decoder per relation.

    Parameters
    ----------
    in_dim       : input feature dimension
    hidden_dim   : hidden (and output embedding) dimension
    n_relations  : number of side effect types
    n_bases      : basis matrices for decomposition
    n_layers     : number of RGCN layers
    dropout      : dropout rate
    """

    def __init__(self, in_dim, hidden_dim, n_relations, n_bases, n_layers=2,
                 dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * n_layers
        for i in range(n_layers):
            self.layers.append(
                RGCNLayer(dims[i], dims[i+1], n_relations, n_bases, dropout)
            )

        # Bilinear decoder: one weight matrix per relation (via basis decomp)
        self.dec_V = nn.Parameter(torch.FloatTensor(n_bases, hidden_dim, hidden_dim))
        self.dec_A = nn.Parameter(torch.FloatTensor(n_relations, n_bases))
        nn.init.xavier_uniform_(self.dec_V.data.view(n_bases, -1).unsqueeze(0)).squeeze()
        nn.init.xavier_uniform_(self.dec_A.unsqueeze(0)).squeeze()

        self.n_relations = n_relations
        self.dropout     = nn.Dropout(dropout)

    def encode(self, x, edge_index, edge_type):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, edge_type)
        return h  # (n_nodes, hidden_dim)

    def decode(self, z_i, z_j, rel_ids):
        """
        z_i, z_j : (n, hidden_dim)
        rel_ids  : (n,) long tensor of relation indices
        Returns  : (n,) logit scores
        """
        scores = torch.zeros(len(z_i), device=z_i.device)
        unique_rels = rel_ids.unique()
        for r in unique_rels:
            mask = (rel_ids == r)
            W_r  = torch.einsum("b,bde->de", self.dec_A[r], self.dec_V)
            zi_r = self.dropout(z_i[mask])
            zj_r = self.dropout(z_j[mask])
            scores[mask] = (zi_r @ W_r * zj_r).sum(dim=1)
        return scores

    def forward(self, x, edge_index, edge_type, src_idx, dst_idx, rel_ids):
        z     = self.encode(x, edge_index, edge_type)
        z_i   = z[src_idx]
        z_j   = z[dst_idx]
        return self.decode(z_i, z_j, rel_ids)


# ── High-level predictor ───────────────────────────────────────────────────────

class RGCNPredictor:
    """
    Wraps RGCN training and evaluation.

    Parameters
    ----------
    feature_set : DrugFeatureSet — provides initial drug node features
    drug_list   : list of drug STITCH IDs (defines node ordering)
    se_list     : list of SE CUI strings (defines relation ordering)
    params      : RGCN hyperparameter dict (default from config)
    device      : "cpu" | "cuda"
    """

    def __init__(self, feature_set, drug_list, se_list,
                 params=None, device=None):
        if not TORCH_OK:
            raise RuntimeError("PyTorch is required for RGCNPredictor.")

        self.feature_set = feature_set
        self.drug_list   = list(drug_list)
        self.drug2idx    = {d: i for i, d in enumerate(self.drug_list)}
        self.se_list     = list(se_list)
        self.se2idx      = {s: i for i, s in enumerate(self.se_list)}
        self.params      = params or RGCN_PARAMS.copy()
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.node_features = None  # torch.Tensor

    def _build_node_features(self, feature_mode="target+fp"):
        """Build initial node feature tensor from DrugFeatureSet."""
        X = self.feature_set.get(self.drug_list, mode=feature_mode)
        return torch.FloatTensor(X).to(self.device)

    def _build_graph(self, combo_df):
        """
        Build edge_index and edge_type tensors from combo DataFrame.

        combo_df must have: STITCH 1, STITCH 2, Polypharmacy Side Effect
        Returns (edge_index, edge_type) — both include reverse edges for symmetry.
        """
        src, dst, rtype = [], [], []
        for _, row in combo_df.iterrows():
            d1 = self.drug2idx.get(row["STITCH 1"])
            d2 = self.drug2idx.get(row["STITCH 2"])
            r  = self.se2idx.get(row["Polypharmacy Side Effect"])
            if d1 is None or d2 is None or r is None:
                continue
            src.append(d1); dst.append(d2); rtype.append(r)
            src.append(d2); dst.append(d1); rtype.append(r)  # symmetric

        edge_index = torch.LongTensor([src, dst]).to(self.device)
        edge_type  = torch.LongTensor(rtype).to(self.device)
        return edge_index, edge_type

    def _triples_to_tensors(self, df):
        """Convert a split DataFrame to (src, dst, rel, label) tensors."""
        src   = torch.LongTensor([self.drug2idx[d] for d in df["drug_1"]]).to(self.device)
        dst   = torch.LongTensor([self.drug2idx[d] for d in df["drug_2"]]).to(self.device)
        rel   = torch.LongTensor([self.se2idx[s]   for s in df["side_effect"]]).to(self.device)
        label = torch.FloatTensor(df["label"].values).to(self.device)
        return src, dst, rel, label

    def _edge_sample(self, edge_index, edge_type, n_edges):
        """Random edge sub-sampling for mini-batch training."""
        n_total = edge_index.size(1)
        idx     = torch.randperm(n_total)[:n_edges]
        return edge_index[:, idx], edge_type[idx]

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, combo_df, train_df, val_df,
              feature_mode="target+fp"):
        """
        Train the RGCN.

        combo_df : full filtered combo DataFrame for graph construction
        train_df : training triples DataFrame
        val_df   : validation triples DataFrame
        """
        p = self.params
        torch.manual_seed(p.get("random_state", RANDOM_SEED))

        print(f"  Building graph and features...")
        self.node_features = self._build_node_features(feature_mode)
        edge_index, edge_type = self._build_graph(combo_df)

        in_dim = self.node_features.shape[1]
        self.model = RGCN(
            in_dim      = in_dim,
            hidden_dim  = p["hidden_dim"],
            n_relations = len(self.se_list),
            n_bases     = p["num_bases"],
            n_layers    = p["num_layers"],
            dropout     = p["dropout"],
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=p["lr"], weight_decay=p["weight_decay"]
        )

        tr_src, tr_dst, tr_rel, tr_lab = self._triples_to_tensors(train_df)
        vl_src, vl_dst, vl_rel, vl_lab = self._triples_to_tensors(val_df)

        batch_sz  = p["batch_size"]
        best_val  = -np.inf
        best_wts  = None

        print(f"  Training RGCN for {p['epochs']} epochs...")
        for epoch in range(p["epochs"]):
            self.model.train()
            # Mini-batch loop
            perm = torch.randperm(len(tr_src))
            total_loss = 0.0
            n_batches  = 0

            for start in range(0, len(tr_src), batch_sz):
                idx_b  = perm[start:start + batch_sz]
                s_b, d_b, r_b, y_b = tr_src[idx_b], tr_dst[idx_b], tr_rel[idx_b], tr_lab[idx_b]

                # Sample a sub-graph of edges for this batch
                ei_b, et_b = self._edge_sample(edge_index, edge_type,
                                               min(edge_index.size(1), 50_000))
                logits = self.model(self.node_features, ei_b, et_b, s_b, d_b, r_b)
                loss   = F.binary_cross_entropy_with_logits(logits, y_b)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(self.node_features, edge_index, edge_type,
                                        vl_src, vl_dst, vl_rel)
                val_proba  = torch.sigmoid(val_logits).cpu().numpy()
                val_y      = vl_lab.cpu().numpy()
            val_metrics = compute_metrics(val_y, val_proba)
            val_auroc   = val_metrics["auroc"]

            if val_auroc > best_val:
                best_val = val_auroc
                best_wts = {k: v.clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{p['epochs']}  "
                      f"loss={total_loss/n_batches:.4f}  val_auroc={val_auroc:.4f}")

        self.model.load_state_dict(best_wts)
        print(f"  Best val AUROC: {best_val:.4f}")

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def predict(self, combo_df, test_df) -> np.ndarray:
        """Return probability scores for test triples."""
        assert self.model is not None, "Call train() first."
        self.model.eval()
        edge_index, edge_type = self._build_graph(combo_df)
        te_src, te_dst, te_rel, _ = self._triples_to_tensors(test_df)
        with torch.no_grad():
            logits = self.model(self.node_features, edge_index, edge_type,
                                te_src, te_dst, te_rel)
        return torch.sigmoid(logits).cpu().numpy()

    def evaluate_all(self, splits: dict, combo_df,
                     se_ids=None, feature_mode="target+fp") -> pd.DataFrame:
        """
        Evaluate per SE after training.
        Assumes model is already trained (call train() first).
        """
        if se_ids is None:
            se_ids = list(splits.keys())
        rows = []
        edge_index, edge_type = self._build_graph(combo_df)
        self.model.eval()

        for se_id in se_ids:
            split = splits.get(se_id)
            if split is None or len(split.test) == 0:
                continue
            te_src, te_dst, te_rel, te_lab = self._triples_to_tensors(split.test)
            with torch.no_grad():
                logits = self.model(self.node_features, edge_index, edge_type,
                                    te_src, te_dst, te_rel)
            proba   = torch.sigmoid(logits).cpu().numpy()
            y_true  = te_lab.cpu().numpy()
            metrics = compute_metrics(y_true, proba)
            rows.append({"se_id": se_id, "model": "rgcn",
                         "feature_mode": feature_mode,
                         "strategy": split.strategy, **metrics})
        return pd.DataFrame(rows)
