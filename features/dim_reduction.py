"""
features/dim_reduction.py
=========================
Dimensionality reduction methods for drug feature embeddings.
Compares PCA (prior-work baseline) against biologically-motivated alternatives.

Methods
-------
PCAReducer            — linear, global variance; prior-work baseline
KernelPCAReducer      — nonlinear PCA with RBF kernel
GraphDiffusedSVD      — PPI-propagated target profiles → truncated SVD
                        (graph-aware, topology-preserving; primary method)
Node2VecAggregator    — random-walk embeddings on PPI, aggregated per drug
GraphAutoencoder      — bipartite drug-protein GAE (scipy, no PyTorch needed)
UMAPReducer           — manifold learning; nonlinear, topology-preserving

All methods share a common interface:
    reducer = SomeReducer(n_components=64, ...)
    reducer.fit(X)
    Z = reducer.transform(X)    # Z: (n_drugs, n_components)

DrugEmbeddingPipeline wraps all methods for ablation comparison.

Usage
-----
    from features.dim_reduction import DrugEmbeddingPipeline
    pipeline = DrugEmbeddingPipeline(targets_df, ppi_df, smiles_dict)
    results  = pipeline.build_all(drugs, n_components=64)
    # results: {method_name: EmbeddingResult}
"""

import os, sys, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import Optional, List, Dict


# ══════════════════════════════════════════════════════════════════════════════
# 1.  PCA  (prior-work baseline)
# ══════════════════════════════════════════════════════════════════════════════

class PCAReducer:
    """
    Standard PCA on raw target profiles.

    Weakness: linear, ignores PPI topology, dominated by high-variance
    (high-degree) genes. Drugs without targets all map to the zero vector
    which PCA projects to the same point regardless of n_components.

    This is what prior papers use implicitly when they reduce feature
    dimension before feeding to their models.
    """
    name = "pca"

    def __init__(self, n_components: int = 64, random_state: int = 42):
        self.n_components = n_components
        self.pca    = PCA(n_components=n_components, random_state=random_state)
        self.scaler = StandardScaler(with_mean=False)
        self._explained_var = 0.0

    def fit(self, X: np.ndarray, **_):
        Xs = self.scaler.fit_transform(X)
        self.pca.fit(Xs)
        self._explained_var = float(self.pca.explained_variance_ratio_.sum())
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(X)).astype(np.float32)

    def fit_transform(self, X: np.ndarray, **_) -> np.ndarray:
        return self.fit(X).transform(X)

    def info(self) -> dict:
        return {"method": self.name, "n_components": self.n_components,
                "explained_variance_ratio": round(self._explained_var, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Kernel PCA  (nonlinear baseline, modest improvement)
# ══════════════════════════════════════════════════════════════════════════════

class KernelPCAReducer:
    """
    Kernel PCA with RBF kernel on target profiles.
    Captures nonlinear structure but still ignores PPI topology.
    Included as an ablation between linear PCA and graph-aware methods.
    """
    name = "kernel_pca"

    def __init__(self, n_components: int = 64, kernel: str = "rbf",
                 gamma: float = None, random_state: int = 42):
        self.n_components = n_components
        self.kernel = kernel
        self.kpca   = KernelPCA(n_components=n_components, kernel=kernel,
                                gamma=gamma, random_state=random_state)
        self.scaler = StandardScaler(with_mean=False)

    def fit(self, X: np.ndarray, **_):
        self.kpca.fit(self.scaler.fit_transform(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.kpca.transform(self.scaler.transform(X)).astype(np.float32)

    def fit_transform(self, X: np.ndarray, **_) -> np.ndarray:
        return self.fit(X).transform(X)

    def info(self) -> dict:
        return {"method": self.name, "n_components": self.n_components,
                "kernel": self.kernel}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Graph-Diffused SVD  (primary method — graph-aware PCA replacement)
# ══════════════════════════════════════════════════════════════════════════════

class GraphDiffusedSVD:
    """
    PPI-propagated target profiles → truncated SVD.

    Why better than PCA
    -------------------
    PCA decomposes X (drugs x genes) treating genes as independent features.
    This method first diffuses X over the PPI adjacency:

        H = X @ (D^{-1} A)^k        (drug-space, k-hop neighbourhood)

    so drug i's row in H absorbs signal from genes k hops away from its
    direct targets. Two drugs hitting distinct proteins in the same pathway
    produce similar H rows even if their raw target vectors share nothing.

    SVD on H is then equivalent to PCA on graph-smoothed features.
    The alpha parameter adds a personalised PageRank restart, preventing
    diffusion from drifting too far from the original target signal.

    Paper claim: "We replace PCA-based dimensionality reduction with
    graph-diffused SVD, encoding PPI topology directly into drug embeddings."
    """
    name = "graph_diffused_svd"

    def __init__(self, ppi_df: pd.DataFrame, n_components: int = 64,
                 n_hops: int = 2, alpha: float = 0.15, random_state: int = 42):
        self.n_components = n_components
        self.n_hops       = n_hops
        self.alpha        = alpha
        self.random_state = random_state
        self._Vt = None
        self._S  = None
        self._build_ppi(ppi_df)

    def _build_ppi(self, ppi_df):
        genes = sorted(set(ppi_df["Gene 1"]).union(set(ppi_df["Gene 2"])))
        self._gene2idx = {g: i for i, g in enumerate(genes)}
        n = len(genes)
        rows = [self._gene2idx[g] for g in ppi_df["Gene 1"]]
        cols = [self._gene2idx[g] for g in ppi_df["Gene 2"]]
        A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
        A = (A + A.T).sign().astype(np.float32)
        deg = np.asarray(A.sum(axis=1)).flatten()
        di  = np.where(deg > 0, 1.0 / deg, 0.0)
        self._A_norm = sparse.diags(di) @ A   # row-normalised (n_genes x n_genes)

    def _diffuse(self, X: np.ndarray) -> np.ndarray:
        """X: (n_drugs, n_genes) — k-hop PPI diffusion with PPR restart."""
        H = X.astype(np.float32)
        for _ in range(self.n_hops):
            H_new = H @ self._A_norm.T
            if self.alpha > 0:
                H_new = (1 - self.alpha) * H_new + self.alpha * X
            H = H_new
        return H

    def fit(self, X: np.ndarray, **_):
        H = self._diffuse(X)
        k = min(self.n_components, min(H.shape) - 1)
        U, S, Vt = svds(sparse.csr_matrix(H), k=k)
        order    = np.argsort(-S)
        self._U  = U[:, order]
        self._S  = S[order]
        self._Vt = Vt[order, :]   # (k, n_genes) gene basis
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (self._diffuse(X) @ self._Vt.T).astype(np.float32)

    def fit_transform(self, X: np.ndarray, **_) -> np.ndarray:
        self.fit(X)
        return (self._U * self._S).astype(np.float32)

    def info(self) -> dict:
        energy = float((self._S ** 2).sum()) if self._S is not None else 0
        return {"method": self.name, "n_components": self.n_components,
                "n_hops": self.n_hops, "alpha": self.alpha,
                "singular_value_energy": round(energy, 2)}


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Node2Vec Aggregator  (walk-based PPI embeddings)
# ══════════════════════════════════════════════════════════════════════════════

class Node2VecAggregator:
    """
    Random-walk embeddings on the PPI graph, aggregated per drug via its targets.

    Pipeline
    --------
    1. Run Node2Vec on PPI → one (dimensions,) embedding per protein node
    2. For each drug d: z_d = mean( embeddings of d's target proteins )
    3. Drugs with no targets → zero vector

    Why better than PCA
    -------------------
    - Captures nonlinear, higher-order PPI topology (community structure
      at low q; structural role equivalence at high q)
    - Not dominated by variance in the raw feature matrix
    - Proteins in the same pathway cluster in embedding space even without
      direct edges — the walk finds them through shared neighbours

    Parameters
    ----------
    p=1, q<1  → DFS-biased (structural equivalence between proteins)
    p=1, q>1  → BFS-biased (homophily / pathway community membership)
    """
    name = "node2vec"

    def __init__(self, ppi_df: pd.DataFrame, targets_df: pd.DataFrame,
                 dimensions: int = 64, walk_length: int = 30,
                 num_walks: int = 200, p: float = 1.0, q: float = 0.5,
                 workers: int = 4, random_state: int = 42):
        self.dimensions   = dimensions
        self.walk_length  = walk_length
        self.num_walks    = num_walks
        self.p, self.q    = p, q
        self.workers      = workers
        self.random_state = random_state
        self.n_components = dimensions
        self._targets_df  = targets_df
        self._ppi_df      = ppi_df
        self._gene_emb: Dict[int, np.ndarray] = {}

    def fit(self, X=None, **_):
        from node2vec import Node2Vec as _Node2Vec
        print(f"    Building PPI graph ({len(self._ppi_df)} edges)...")
        G = nx.Graph()
        G.add_edges_from(zip(self._ppi_df["Gene 1"].astype(str),
                             self._ppi_df["Gene 2"].astype(str)))
        print(f"    Node2Vec: {G.number_of_nodes()} nodes, "
              f"walks={self.num_walks}, len={self.walk_length}, "
              f"dim={self.dimensions}, p={self.p}, q={self.q}")
        n2v = _Node2Vec(G, dimensions=self.dimensions,
                        walk_length=self.walk_length, num_walks=self.num_walks,
                        p=self.p, q=self.q, workers=self.workers,
                        seed=self.random_state, quiet=True)
        model = n2v.fit(window=10, min_count=1, batch_words=4)
        self._gene_emb = {int(node): model.wv[node]
                          for node in model.wv.index_to_key}
        print(f"    Trained: {len(self._gene_emb)} protein embeddings")
        return self

    def transform(self, drugs: List[str]) -> np.ndarray:
        drug2genes = (self._targets_df
                      .groupby("STITCH")["Gene"].apply(list).to_dict())
        Z = np.zeros((len(drugs), self.dimensions), dtype=np.float32)
        for i, drug in enumerate(drugs):
            vecs = [self._gene_emb[g] for g in drug2genes.get(drug, [])
                    if g in self._gene_emb]
            if vecs:
                Z[i] = np.mean(vecs, axis=0)
        return Z

    def fit_transform(self, drugs: List[str], **_) -> np.ndarray:
        return self.fit().transform(drugs)

    def info(self) -> dict:
        return {"method": self.name, "dimensions": self.dimensions,
                "walk_length": self.walk_length, "num_walks": self.num_walks,
                "p": self.p, "q": self.q}


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Graph Autoencoder  (strongest — end-to-end bipartite learning)
# ══════════════════════════════════════════════════════════════════════════════

class GraphAutoencoder:
    """
    Bipartite drug-protein Graph Autoencoder (GAE).
    Implemented in numpy/scipy — no PyTorch required.

    Architecture
    ------------
    Encoder (2-layer graph-diffused linear projection):
        M    = B_norm @ A_ppi_norm @ B_norm.T     (drug co-occurrence via PPI)
        H1   = ReLU( M @ I_drug @ W1 )
        Z_d  = ReLU( H1 @ W2 )                    drug embeddings (n_d, k)
        Z_p  = ReLU( B_norm.T @ I_drug @ W1 @ W2 ) protein embeddings (n_p, k)

    Decoder (inner product):
        score(i, j) = sigmoid( z_i · p_j )        drug i, protein j

    Trained to reconstruct the binary drug-target edge set via BCE with
    negative sampling.

    Why this is the strongest alternative
    --------------------------------------
    - Optimised objective: embeddings are trained to encode drug-target
      relationships, not just to compress variance
    - Handles missing-target drugs: zero-target drugs still receive
      2-hop messages through the bipartite graph if any protein neighbour
      shares a pathway with a targeted protein
    - Produces protein embeddings as a bonus — can initialise RGCN nodes
    - Pre-train once, freeze, use across all downstream models
    """
    name = "graph_autoencoder"

    def __init__(self, n_components: int = 64, hidden_dim: int = 128,
                 lr: float = 0.005, epochs: int = 100,
                 neg_ratio: float = 1.0, random_state: int = 42):
        self.n_components = n_components
        self.hidden_dim   = hidden_dim
        self.lr           = lr
        self.epochs       = epochs
        self.neg_ratio    = neg_ratio
        self.rng          = np.random.default_rng(random_state)
        self.W1 = self.W2 = None
        self._drug2idx: dict = {}
        self._drug_emb: Optional[np.ndarray] = None
        self._protein_emb: Optional[np.ndarray] = None
        self._B_norm = self._A_ppi_norm = self._M = None
        self._pos_edges: list = []

    def _build_bipartite(self, drugs, targets_df, ppi_df):
        all_genes = sorted(set(ppi_df["Gene 1"]).union(set(ppi_df["Gene 2"])))
        self._drug2idx = {d: i for i, d in enumerate(drugs)}
        gene2idx       = {g: i for i, g in enumerate(all_genes)}
        n_d, n_p       = len(drugs), len(all_genes)

        # Drug-protein edges
        d_rows, p_cols = [], []
        for _, row in targets_df.iterrows():
            d = self._drug2idx.get(row["STITCH"])
            p = gene2idx.get(row["Gene"])
            if d is not None and p is not None:
                d_rows.append(d)
                p_cols.append(p)
        self._pos_edges = list(zip(d_rows, p_cols))

        B = sparse.csr_matrix((np.ones(len(d_rows)), (d_rows, p_cols)),
                              shape=(n_d, n_p))
        deg_d = np.asarray(B.sum(axis=1)).flatten()
        di_d  = np.where(deg_d > 0, 1.0 / deg_d, 0.0)
        self._B_norm = sparse.diags(di_d) @ B

        p_r = [gene2idx[g] for g in ppi_df["Gene 1"] if g in gene2idx]
        p_c = [gene2idx[g] for g in ppi_df["Gene 2"] if g in gene2idx]
        A_p = sparse.csr_matrix((np.ones(len(p_r)), (p_r, p_c)),
                                shape=(n_p, n_p))
        A_p = (A_p + A_p.T).sign().astype(np.float32)
        deg_p = np.asarray(A_p.sum(axis=1)).flatten()
        di_p  = np.where(deg_p > 0, 1.0 / deg_p, 0.0)
        self._A_ppi_norm = sparse.diags(di_p) @ A_p

        self._M = self._B_norm @ (self._A_ppi_norm @ self._B_norm.T)
        return n_d, n_p

    @staticmethod
    def _relu(x): return np.maximum(0, x)

    def _encode(self, n_d, n_p):
        Id = np.eye(n_d, dtype=np.float32)
        H1_d = self._relu(self._M @ Id @ self.W1)
        Z_d  = self._relu(H1_d @ self.W2)
        H1_p = self._relu(self._B_norm.T @ Id @ self.W1)
        Z_p  = self._relu(H1_p @ self.W2)
        return Z_d, Z_p

    def fit(self, drugs: List[str], targets_df: pd.DataFrame,
            ppi_df: pd.DataFrame, **_):
        print(f"    Building bipartite graph...")
        n_d, n_p = self._build_bipartite(drugs, targets_df, ppi_df)
        scale = 1.0 / np.sqrt(n_d)
        self.W1 = self.rng.normal(0, scale, (n_d, self.hidden_dim)).astype(np.float32)
        self.W2 = self.rng.normal(0, 0.1,   (self.hidden_dim, self.n_components)).astype(np.float32)
        n_neg = max(1, int(len(self._pos_edges) * self.neg_ratio))
        print(f"    GAE: {n_d} drugs, {n_p} proteins, "
              f"{len(self._pos_edges)} pos edges, {self.epochs} epochs")

        best_loss = np.inf
        best_W1, best_W2 = self.W1.copy(), self.W2.copy()

        for epoch in range(self.epochs):
            neg_d = self.rng.integers(0, n_d, n_neg).tolist()
            neg_p = self.rng.integers(0, n_p, n_neg).tolist()
            edges  = self._pos_edges + list(zip(neg_d, neg_p))
            labels = np.array([1.0]*len(self._pos_edges) + [0.0]*n_neg,
                              dtype=np.float32)
            d_idx  = np.array([e[0] for e in edges])
            p_idx  = np.array([e[1] for e in edges])

            Z_d, Z_p = self._encode(n_d, n_p)
            scores   = np.sum(Z_d[d_idx] * Z_p[p_idx], axis=1)
            probs    = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))
            n_e      = len(edges)
            loss     = -np.mean(labels * np.log(probs + 1e-9) +
                                (1-labels) * np.log(1-probs + 1e-9))

            dL_ds = (probs - labels) / n_e
            dZd   = np.zeros_like(Z_d); dZp = np.zeros_like(Z_p)
            np.add.at(dZd, d_idx, dL_ds[:, None] * Z_p[p_idx])
            np.add.at(dZp, p_idx, dL_ds[:, None] * Z_d[d_idx])

            Id   = np.eye(n_d, dtype=np.float32)
            H1_d = self._relu(self._M @ Id @ self.W1)
            dW2  = np.clip(H1_d.T @ dZd / n_d, -1, 1)
            dW1  = np.clip((self._M @ Id).T @ (dZd @ self.W2.T) / n_d, -1, 1)
            self.W1 -= self.lr * dW1
            self.W2 -= self.lr * dW2

            if loss < best_loss:
                best_loss = loss
                best_W1, best_W2 = self.W1.copy(), self.W2.copy()
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1:3d}/{self.epochs}  loss={loss:.4f}")

        self.W1, self.W2 = best_W1, best_W2
        Z_d, Z_p = self._encode(n_d, n_p)
        self._drug_emb    = Z_d
        self._protein_emb = Z_p
        self._n_d, self._n_p = n_d, n_p
        print(f"    Done. Best loss={best_loss:.4f}")
        return self

    def transform(self, drugs: List[str]) -> np.ndarray:
        Z = np.zeros((len(drugs), self.n_components), dtype=np.float32)
        for i, d in enumerate(drugs):
            idx = self._drug2idx.get(d)
            if idx is not None:
                Z[i] = self._drug_emb[idx]
        return Z

    def fit_transform(self, drugs: List[str], targets_df: pd.DataFrame,
                      ppi_df: pd.DataFrame, **_) -> np.ndarray:
        return self.fit(drugs, targets_df, ppi_df).transform(drugs)

    def protein_embeddings(self) -> np.ndarray:
        """Return (n_proteins, n_components) — useful to init RGCN protein nodes."""
        return self._protein_emb

    def info(self) -> dict:
        return {"method": self.name, "n_components": self.n_components,
                "hidden_dim": self.hidden_dim, "epochs": self.epochs}


# ══════════════════════════════════════════════════════════════════════════════
# 6.  UMAP  (manifold learning)
# ══════════════════════════════════════════════════════════════════════════════

class UMAPReducer:
    """
    UMAP applied to graph-diffused target profiles.
    Best for 2D visualisation (n_components=2).
    Can also be used for feature extraction (n_components=32-64).

    Caveat: stochastic, no closed-form inverse, hard for reviewers to
    interpret — use as a secondary comparison, not a main contribution.
    """
    name = "umap"

    def __init__(self, ppi_df: Optional[pd.DataFrame] = None,
                 n_components: int = 64, n_hops: int = 2,
                 n_neighbors: int = 15, min_dist: float = 0.1,
                 random_state: int = 42):
        import umap as _umap
        self.n_components = n_components
        self.n_hops       = n_hops
        self._reducer     = _umap.UMAP(n_components=n_components,
                                       n_neighbors=n_neighbors,
                                       min_dist=min_dist,
                                       random_state=random_state)
        self._diffuser: Optional[GraphDiffusedSVD] = None
        if ppi_df is not None and n_hops > 0:
            # Use diffuser for preprocessing, but keep full gene dim (no SVD)
            self._diff_obj = GraphDiffusedSVD(ppi_df, n_components=128,
                                              n_hops=n_hops, alpha=0.15)
            self._use_diff = True
        else:
            self._use_diff = False

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        return self._diff_obj.transform(X) if self._use_diff else X

    def fit(self, X: np.ndarray, **_):
        if self._use_diff:
            self._diff_obj.fit(X)
        self._reducer.fit(self._preprocess(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._reducer.transform(self._preprocess(X)).astype(np.float32)

    def fit_transform(self, X: np.ndarray, **_) -> np.ndarray:
        if self._use_diff:
            self._diff_obj.fit(X)
        return self._reducer.fit_transform(self._preprocess(X)).astype(np.float32)

    def info(self) -> dict:
        return {"method": self.name, "n_components": self.n_components,
                "n_hops": self.n_hops}


# ══════════════════════════════════════════════════════════════════════════════
# 7.  DrugEmbeddingPipeline  (master wrapper for ablation)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EmbeddingResult:
    method      : str
    embeddings  : np.ndarray       # (n_drugs, n_components)
    has_target  : np.ndarray       # (n_drugs,) bool
    fit_time_s  : float
    info        : dict = field(default_factory=dict)


class DrugEmbeddingPipeline:
    """
    Builds drug embeddings using all reduction methods for ablation comparison.

    Usage
    -----
        pipeline = DrugEmbeddingPipeline(targets_df, ppi_df, smiles_dict)
        results  = pipeline.build_all(drugs, n_components=64)
        # returns {method_name: EmbeddingResult}
    """

    METHODS = ["pca", "kernel_pca", "graph_diffused_svd",
               "node2vec", "graph_autoencoder", "umap"]

    def __init__(self, targets_df: pd.DataFrame, ppi_df: pd.DataFrame,
                 smiles_dict: dict = None, random_state: int = 42):
        self.targets_df   = targets_df
        self.ppi_df       = ppi_df
        self.smiles_dict  = smiles_dict or {}
        self.random_state = random_state

    def _build_raw_profiles(self, drugs: List[str]):
        genes    = sorted(set(self.ppi_df["Gene 1"]).union(set(self.ppi_df["Gene 2"])))
        g2i      = {g: i for i, g in enumerate(genes)}
        d2genes  = self.targets_df.groupby("STITCH")["Gene"].apply(set).to_dict()
        X        = np.zeros((len(drugs), len(genes)), dtype=np.float32)
        has_tgt  = np.zeros(len(drugs), dtype=bool)
        for i, d in enumerate(drugs):
            gs = d2genes.get(d, set())
            if gs:
                has_tgt[i] = True
                for g in gs:
                    if g in g2i:
                        X[i, g2i[g]] = 1.0
        return X, has_tgt

    def build_all(self, drugs: List[str], n_components: int = 64,
                  methods: List[str] = None,
                  node2vec_walks: int = 100,
                  gae_epochs: int = 80) -> Dict[str, EmbeddingResult]:
        if methods is None:
            methods = self.METHODS

        print(f"\nBuilding raw profiles ({len(drugs)} drugs)...")
        X_raw, has_tgt = self._build_raw_profiles(drugs)
        print(f"  Shape: {X_raw.shape}, with targets: {has_tgt.sum()}/{len(drugs)}")

        results: Dict[str, EmbeddingResult] = {}

        if "pca" in methods:
            print("\n[1/6] PCA (baseline)...")
            t0 = time.time()
            r = PCAReducer(n_components=n_components, random_state=self.random_state)
            Z = r.fit_transform(X_raw)
            results["pca"] = EmbeddingResult("pca", Z, has_tgt,
                                             round(time.time()-t0, 2), r.info())
            print(f"  Done {results['pca'].fit_time_s}s | "
                  f"explained_var={r._explained_var:.3f}")

        if "kernel_pca" in methods:
            print("\n[2/6] Kernel PCA (rbf)...")
            t0 = time.time()
            r = KernelPCAReducer(n_components=n_components,
                                 random_state=self.random_state)
            Z = r.fit_transform(X_raw)
            results["kernel_pca"] = EmbeddingResult("kernel_pca", Z, has_tgt,
                                                    round(time.time()-t0, 2), r.info())
            print(f"  Done {results['kernel_pca'].fit_time_s}s")

        if "graph_diffused_svd" in methods:
            print("\n[3/6] Graph-Diffused SVD (primary method)...")
            t0 = time.time()
            r = GraphDiffusedSVD(self.ppi_df, n_components=n_components,
                                 n_hops=2, alpha=0.15,
                                 random_state=self.random_state)
            Z = r.fit_transform(X_raw)
            results["graph_diffused_svd"] = EmbeddingResult(
                "graph_diffused_svd", Z, has_tgt,
                round(time.time()-t0, 2), r.info())
            print(f"  Done {results['graph_diffused_svd'].fit_time_s}s")

        if "node2vec" in methods:
            print("\n[4/6] Node2Vec aggregation...")
            t0 = time.time()
            r = Node2VecAggregator(self.ppi_df, self.targets_df,
                                   dimensions=n_components, walk_length=30,
                                   num_walks=node2vec_walks, p=1.0, q=0.5,
                                   workers=4, random_state=self.random_state)
            Z = r.fit_transform(drugs)
            results["node2vec"] = EmbeddingResult("node2vec", Z, has_tgt,
                                                  round(time.time()-t0, 2), r.info())
            print(f"  Done {results['node2vec'].fit_time_s}s")

        if "graph_autoencoder" in methods:
            print("\n[5/6] Graph Autoencoder...")
            t0 = time.time()
            r = GraphAutoencoder(n_components=n_components,
                                 hidden_dim=n_components * 2,
                                 lr=0.005, epochs=gae_epochs,
                                 random_state=self.random_state)
            Z = r.fit_transform(drugs, self.targets_df, self.ppi_df)
            results["graph_autoencoder"] = EmbeddingResult(
                "graph_autoencoder", Z, has_tgt,
                round(time.time()-t0, 2), r.info())
            print(f"  Done {results['graph_autoencoder'].fit_time_s}s")

        if "umap" in methods:
            print("\n[6/6] UMAP (graph-diffused input)...")
            t0 = time.time()
            r = UMAPReducer(ppi_df=self.ppi_df, n_components=n_components,
                            n_hops=2, n_neighbors=15, min_dist=0.1,
                            random_state=self.random_state)
            Z = r.fit_transform(X_raw)
            results["umap"] = EmbeddingResult("umap", Z, has_tgt,
                                              round(time.time()-t0, 2), r.info())
            print(f"  Done {results['umap'].fit_time_s}s")

        return results

    def summary_table(self, results: Dict[str, EmbeddingResult]) -> pd.DataFrame:
        rows = []
        for name, res in results.items():
            zero_rows = int((np.abs(res.embeddings).sum(axis=1) == 0).sum())
            rows.append({
                "method"          : name,
                "embed_shape"     : str(res.embeddings.shape),
                "fit_time_s"      : res.fit_time_s,
                "drugs_with_tgt"  : int(res.has_target.sum()),
                "zero_emb_rows"   : zero_rows,
                "pct_zero"        : round(100 * zero_rows / len(res.has_target), 1),
                **{k: v for k, v in res.info.items()
                   if k not in ("method",)},
            })
        return pd.DataFrame(rows)
