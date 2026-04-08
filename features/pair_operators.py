"""
features/pair_operators.py
==========================
Symmetric drug pair representation operators.

Given drug embeddings z_i and z_j, each operator produces a fixed-size
vector that is:
  - Permutation-invariant  (f(z_i, z_j) == f(z_j, z_i))
  - Side-effect-agnostic   (the SE relation is handled by the model head)

Available operators
-------------------
  hadamard         : z_i ⊙ z_j
  absdiff          : |z_i − z_j|
  hadamard_absdiff : [z_i ⊙ z_j || |z_i − z_j|]   ← recommended default
  concat_sym       : [z_i || z_j] + [z_j || z_i] (averaged → same dim as concat)
  sum              : z_i + z_j
  concat           : [z_i || z_j]  (NOT symmetric — use only as ablation)

Usage
-----
    from features.pair_operators import make_pair_features, OPERATORS
    X_pair = make_pair_features(z_i, z_j, operator="hadamard_absdiff")
"""

import numpy as np
from typing import Literal

# ── Individual operators ───────────────────────────────────────────────────────

def hadamard(zi: np.ndarray, zj: np.ndarray) -> np.ndarray:
    """Element-wise product. Shape: (n, d)"""
    return zi * zj


def absdiff(zi: np.ndarray, zj: np.ndarray) -> np.ndarray:
    """Absolute element-wise difference. Shape: (n, d)"""
    return np.abs(zi - zj)


def hadamard_absdiff(zi: np.ndarray, zj: np.ndarray) -> np.ndarray:
    """Concatenation of Hadamard product and absolute difference. Shape: (n, 2d)
    Captures both co-activation and divergence signals simultaneously.
    Permutation-invariant. Recommended for all experiments."""
    return np.concatenate([hadamard(zi, zj), absdiff(zi, zj)], axis=1)


def concat_sym(zi: np.ndarray, zj: np.ndarray) -> np.ndarray:
    """Symmetric concatenation: average of [zi||zj] and [zj||zi]. Shape: (n, 2d)
    Enforces permutation invariance unlike raw concatenation."""
    fwd = np.concatenate([zi, zj], axis=1)
    rev = np.concatenate([zj, zi], axis=1)
    return (fwd + rev) / 2.0


def sum_op(zi: np.ndarray, zj: np.ndarray) -> np.ndarray:
    """Element-wise sum. Shape: (n, d). Symmetric but lossy."""
    return zi + zj


def concat(zi: np.ndarray, zj: np.ndarray) -> np.ndarray:
    """Raw concatenation. Shape: (n, 2d). NOT symmetric — ablation only."""
    return np.concatenate([zi, zj], axis=1)


# ── Registry ───────────────────────────────────────────────────────────────────

OPERATORS = {
    "hadamard"        : hadamard,
    "absdiff"         : absdiff,
    "hadamard_absdiff": hadamard_absdiff,
    "concat_sym"      : concat_sym,
    "sum"             : sum_op,
    "concat"          : concat,
}

OUTPUT_DIM_MULTIPLIER = {
    "hadamard"        : 1,
    "absdiff"         : 1,
    "hadamard_absdiff": 2,
    "concat_sym"      : 2,
    "sum"             : 1,
    "concat"          : 2,
}


# ── Main API ───────────────────────────────────────────────────────────────────

def make_pair_features(
    zi: np.ndarray,
    zj: np.ndarray,
    operator: str = "hadamard_absdiff",
) -> np.ndarray:
    """
    Compute pairwise drug features for a batch of drug pairs.

    Parameters
    ----------
    zi       : (n, d) array — embeddings for drug i in each pair
    zj       : (n, d) array — embeddings for drug j in each pair
    operator : one of OPERATORS keys

    Returns
    -------
    np.ndarray of shape (n, d') where d' depends on the operator
    """
    if operator not in OPERATORS:
        raise ValueError(f"Unknown operator '{operator}'. "
                         f"Choose from: {list(OPERATORS)}")
    fn = OPERATORS[operator]
    return fn(zi, zj).astype(np.float32)


def output_dim(drug_dim: int, operator: str) -> int:
    """Return the output dimension for a given drug embedding dim and operator."""
    return drug_dim * OUTPUT_DIM_MULTIPLIER[operator]


def compare_operators(
    zi: np.ndarray,
    zj: np.ndarray,
    operators: list = None,
) -> dict:
    """
    Compute pair features under all (or specified) operators.

    Returns dict {operator_name: feature_matrix}. Useful for ablation loops.
    """
    if operators is None:
        operators = list(OPERATORS)
    return {op: make_pair_features(zi, zj, op) for op in operators}


# ── Bilinear decoder for relational models ─────────────────────────────────────

class BilinearDecoder:
    """
    Relation-specific bilinear scoring: score(i, r, j) = z_i^T W_r z_j
    where W_r = sum_b a_{r,b} * V_b  (basis decomposition).

    Used in the RGCN model head. This class holds the numpy weights only;
    the torch version is in models/rgcn.py.

    Parameters
    ----------
    n_relations  : number of side effect types (e.g., 963)
    embedding_dim: drug embedding dimension
    n_bases      : number of shared basis matrices
    """

    def __init__(self, n_relations: int, embedding_dim: int, n_bases: int = 32,
                 rng: np.random.Generator = None):
        rng = rng or np.random.default_rng(42)
        scale = 1.0 / np.sqrt(embedding_dim)
        self.V = rng.normal(0, scale, (n_bases, embedding_dim, embedding_dim)).astype(np.float32)
        self.A = rng.normal(0, 0.1, (n_relations, n_bases)).astype(np.float32)
        self.n_relations   = n_relations
        self.embedding_dim = embedding_dim

    def score(self, zi: np.ndarray, zj: np.ndarray, relation_ids: np.ndarray) -> np.ndarray:
        """
        Compute bilinear scores for a batch of (drug_i, drug_j, relation) triples.

        zi, zj        : (n, d)
        relation_ids  : (n,) integer relation indices

        Returns (n,) float scores.
        """
        scores = np.zeros(len(zi), dtype=np.float32)
        unique_rels = np.unique(relation_ids)
        for r in unique_rels:
            mask = relation_ids == r
            W_r = np.einsum("b,bde->de", self.A[r], self.V)  # (d, d)
            zi_r = zi[mask]   # (n_r, d)
            zj_r = zj[mask]   # (n_r, d)
            scores[mask] = np.sum((zi_r @ W_r) * zj_r, axis=1)
        return scores
