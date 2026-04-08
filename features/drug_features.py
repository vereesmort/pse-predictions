"""
features/drug_features.py
=========================
Drug-level feature computation:
  1. Protein target profiles  — binary vector over PPI gene universe
  2. PPI neighbourhood embeddings — k-hop propagation over the PPI graph
  3. Morgan fingerprints (ECFP4)  — structural/chemical representation
  4. Physicochemical descriptors  — RDKit molecular properties

All features are returned as numpy arrays aligned to a shared drug index.
Missing SMILES → zero fingerprint vector (flagged via `has_smiles` mask).
Missing targets → zero target vector (flagged via `has_target` mask).

Usage:
    from features.drug_features import DrugFeatureBuilder
    builder = DrugFeatureBuilder(targets_df, ppi_df)
    features = builder.build_all(drugs, smiles_dict)   # dict: STITCH -> SMILES
"""

import numpy as np
import pandas as pd
from scipy import sparse
import warnings
warnings.filterwarnings("ignore")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("[drug_features] RDKit not available — fingerprint features disabled.")


class DrugFeatureBuilder:
    """
    Builds per-drug feature matrices.

    Parameters
    ----------
    targets_df : pd.DataFrame  with columns [STITCH, Gene]
    ppi_df     : pd.DataFrame  with columns [Gene 1, Gene 2]
    morgan_radius : int
    morgan_nbits  : int
    ppi_hops      : int   neighbourhood hops for propagation
    """

    def __init__(self, targets_df, ppi_df,
                 morgan_radius=2, morgan_nbits=2048, ppi_hops=2):
        self.targets_df    = targets_df
        self.ppi_df        = ppi_df
        self.morgan_radius = morgan_radius
        self.morgan_nbits  = morgan_nbits
        self.ppi_hops      = ppi_hops

        # Build gene universe and adjacency matrix once
        self._build_ppi_graph()

    # ── PPI graph ──────────────────────────────────────────────────────────────
    def _build_ppi_graph(self):
        genes = sorted(set(self.ppi_df["Gene 1"]).union(self.ppi_df["Gene 2"]))
        self.gene2idx = {g: i for i, g in enumerate(genes)}
        self.n_genes  = len(genes)

        rows = [self.gene2idx[g] for g in self.ppi_df["Gene 1"]]
        cols = [self.gene2idx[g] for g in self.ppi_df["Gene 2"]]
        data = np.ones(len(rows), dtype=np.float32)
        A = sparse.csr_matrix((data, (rows, cols)),
                              shape=(self.n_genes, self.n_genes))
        # Symmetrise and row-normalise
        A = A + A.T
        A = (A > 0).astype(np.float32)
        deg = np.asarray(A.sum(axis=1)).flatten()
        deg_inv = np.where(deg > 0, 1.0 / deg, 0.0)
        D_inv = sparse.diags(deg_inv)
        self.A_norm = D_inv @ A  # row-normalised adjacency

    # ── Target profiles ───────────────────────────────────────────────────────
    def build_target_profiles(self, drugs):
        """
        Binary vector over gene universe for each drug.
        Returns (n_drugs × n_genes) float32 array and has_target mask.
        """
        drug2genes = self.targets_df.groupby("STITCH")["Gene"].apply(set).to_dict()
        X = np.zeros((len(drugs), self.n_genes), dtype=np.float32)
        has_target = np.zeros(len(drugs), dtype=bool)
        for i, d in enumerate(drugs):
            genes = drug2genes.get(d, set())
            if genes:
                has_target[i] = True
                for g in genes:
                    if g in self.gene2idx:
                        X[i, self.gene2idx[g]] = 1.0
        return X, has_target

    # ── PPI neighbourhood embeddings ──────────────────────────────────────────
    def build_ppi_embeddings(self, drugs):
        """
        Propagate target profiles over PPI for ppi_hops steps.
        Returns (n_drugs × n_genes) float32 — smoothed target signal.
        """
        X, has_target = self.build_target_profiles(drugs)
        # Propagate: H shape (n_drugs × n_genes), A_norm shape (n_genes × n_genes)
        # H @ A_norm.T  →  (n_drugs × n_genes) ✓
        H = X.copy()
        for _ in range(self.ppi_hops):
            H = H @ self.A_norm.T   # (n_drugs × n_genes) × (n_genes × n_genes)
        return H.astype(np.float32), has_target

    # ── Compact PPI embedding via SVD ─────────────────────────────────────────
    def build_ppi_svd_embeddings(self, drugs, n_components=64):
        """
        Low-rank SVD of the propagated target profile matrix.
        Returns (n_drugs × n_components) — compact, dense embedding.
        """
        from scipy.sparse.linalg import svds
        H, has_target = self.build_ppi_embeddings(drugs)
        k = min(n_components, H.shape[1] - 1, H.shape[0] - 1)
        if H.shape[0] < k + 1:
            return H[:, :k], has_target
        # Only decompose drugs with at least one target
        U, S, Vt = svds(H, k=k)
        emb = U * S  # (n_drugs × k)
        return emb.astype(np.float32), has_target

    # ── Morgan fingerprints ───────────────────────────────────────────────────
    def build_morgan_fingerprints(self, drugs, smiles_dict):
        """
        ECFP4 fingerprints + physicochemical descriptors.

        smiles_dict : {STITCH_id: canonical_SMILES}
        Returns (n_drugs × (morgan_nbits + n_physchem)) array and has_smiles mask.
        """
        if not RDKIT_OK:
            print("[build_morgan_fingerprints] RDKit unavailable — returning zeros.")
            return np.zeros((len(drugs), self.morgan_nbits), dtype=np.float32), \
                   np.zeros(len(drugs), dtype=bool)

        physchem_fns = [
            ("MW",    Descriptors.MolWt),
            ("LogP",  Descriptors.MolLogP),
            ("HBD",   rdMolDescriptors.CalcNumHBD),
            ("HBA",   rdMolDescriptors.CalcNumHBA),
            ("TPSA",  Descriptors.TPSA),
            ("RotB",  rdMolDescriptors.CalcNumRotatableBonds),
            ("Rings", rdMolDescriptors.CalcNumRings),
        ]
        n_physchem = len(physchem_fns)
        total_dim  = self.morgan_nbits + n_physchem
        X          = np.zeros((len(drugs), total_dim), dtype=np.float32)
        has_smiles = np.zeros(len(drugs), dtype=bool)

        for i, drug in enumerate(drugs):
            smi = smiles_dict.get(drug)
            if smi is None:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            # Morgan fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.morgan_radius, nBits=self.morgan_nbits)
            X[i, :self.morgan_nbits] = np.frombuffer(fp.ToBitString().encode(),
                                                      dtype=np.uint8) - ord('0')
            # Physicochemical descriptors (normalised by fixed scales)
            scales = [500, 5, 10, 10, 150, 10, 6]
            for j, (_, fn) in enumerate(physchem_fns):
                try:
                    val = fn(mol)
                    X[i, self.morgan_nbits + j] = float(val) / scales[j]
                except Exception:
                    pass
            has_smiles[i] = True

        return X, has_smiles

    # ── Fetch SMILES from PubChem (fallback) ──────────────────────────────────
    @staticmethod
    def fetch_smiles_pubchem(stitch_ids, cache_path="data/smiles_cache.csv"):
        """
        Convert STITCH CID → PubChem CID → canonical SMILES.
        Caches results to avoid repeated HTTP calls.

        Returns dict {STITCH_id: SMILES or None}
        """
        import os, urllib.request, json, time

        cache = {}
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            cache = dict(zip(df["STITCH"], df["SMILES"]))
        results = {}
        new_fetches = []
        for sid in stitch_ids:
            if sid in cache:
                results[sid] = cache[sid] if pd.notna(cache[sid]) else None
                continue
            # STITCH CID format: "CID000XXXXXX" → PubChem CID = int(XXXXXX)
            try:
                cid = int(sid.replace("CID", "").lstrip("0") or "0")
                url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                       f"cid/{cid}/property/CanonicalSMILES/JSON")
                with urllib.request.urlopen(url, timeout=10) as r:
                    data = json.loads(r.read())
                smi = data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
                results[sid] = smi
                new_fetches.append({"STITCH": sid, "SMILES": smi})
            except Exception:
                results[sid] = None
                new_fetches.append({"STITCH": sid, "SMILES": None})
            time.sleep(0.1)  # polite rate limit

        # Update cache
        if new_fetches:
            new_df = pd.DataFrame(new_fetches)
            if os.path.exists(cache_path):
                old_df = pd.read_csv(cache_path)
                combined = pd.concat([old_df, new_df]).drop_duplicates("STITCH")
            else:
                combined = new_df
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            combined.to_csv(cache_path, index=False)

        return results

    # ── Master build ──────────────────────────────────────────────────────────
    def build_all(self, drugs, smiles_dict=None, ppi_embedding_dim=64):
        """
        Build all features and return a DrugFeatureSet.

        drugs       : list of STITCH IDs (defines the index order)
        smiles_dict : {STITCH: SMILES} or None
        """
        drug_index = list(drugs)
        print(f"  Building target profiles for {len(drug_index)} drugs...")
        target_profiles, has_target = self.build_target_profiles(drug_index)

        print(f"  Building PPI SVD embeddings (dim={ppi_embedding_dim})...")
        ppi_emb, _ = self.build_ppi_svd_embeddings(drug_index, n_components=ppi_embedding_dim)

        if smiles_dict is not None and RDKIT_OK:
            print("  Building Morgan fingerprints...")
            fingerprints, has_smiles = self.build_morgan_fingerprints(drug_index, smiles_dict)
        else:
            fingerprints = np.zeros((len(drug_index), self.morgan_nbits), dtype=np.float32)
            has_smiles   = np.zeros(len(drug_index), dtype=bool)

        return DrugFeatureSet(
            drug_index      = drug_index,
            target_profiles = target_profiles,
            ppi_embeddings  = ppi_emb,
            fingerprints    = fingerprints,
            has_target      = has_target,
            has_smiles      = has_smiles,
        )


class DrugFeatureSet:
    """
    Container for all drug-level features, supporting feature mode selection.
    """

    def __init__(self, drug_index, target_profiles, ppi_embeddings,
                 fingerprints, has_target, has_smiles):
        self.drug_index      = list(drug_index)
        self.drug2idx        = {d: i for i, d in enumerate(self.drug_index)}
        self.target_profiles = target_profiles   # (n, n_genes)
        self.ppi_embeddings  = ppi_embeddings    # (n, ppi_dim)
        self.fingerprints    = fingerprints      # (n, morgan_nbits + physchem)
        self.has_target      = has_target        # (n,) bool
        self.has_smiles      = has_smiles        # (n,) bool

    def get(self, drug_ids, mode="all"):
        """
        Retrieve feature vectors for a list of drug IDs.

        mode options:
          "target_only"   — PPI embeddings only (zero for no-target drugs)
          "fp_only"       — Morgan fingerprints only
          "target+fp"     — PPI emb || fingerprint (primary experimental mode)
          "all"           — target_profiles || ppi_emb || fingerprints
        """
        idx = [self.drug2idx[d] for d in drug_ids]
        tp  = self.target_profiles[idx]
        pe  = self.ppi_embeddings[idx]
        fp  = self.fingerprints[idx]

        if mode == "target_only":
            return pe
        elif mode == "fp_only":
            return fp
        elif mode == "target+fp":
            return np.concatenate([pe, fp], axis=1)
        elif mode == "all":
            return np.concatenate([tp, pe, fp], axis=1)
        else:
            raise ValueError(f"Unknown feature mode: {mode}")

    def feature_dim(self, mode="target+fp"):
        dummy = self.get(self.drug_index[:1], mode=mode)
        return dummy.shape[1]

    def coverage_report(self):
        n = len(self.drug_index)
        print(f"  Total drugs       : {n}")
        print(f"  With targets      : {self.has_target.sum()} ({100*self.has_target.mean():.1f}%)")
        print(f"  Without targets   : {(~self.has_target).sum()} ({100*(~self.has_target).mean():.1f}%)")
        print(f"  With SMILES       : {self.has_smiles.sum()} ({100*self.has_smiles.mean():.1f}%)")
