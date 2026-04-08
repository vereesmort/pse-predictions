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
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
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
            gen = GetMorganGenerator(radius=self.morgan_radius,
                                     fpSize=self.morgan_nbits)
            fp = gen.GetFingerprint(mol)
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

    # ── Fetch SMILES from PubChem ─────────────────────────────────────────────
    @staticmethod
    def fetch_smiles_pubchem(stitch_ids, cache_path="data/smiles_cache.csv",
                             batch_size=100, sleep_batch=0.4, sleep_single=0.25,
                             max_retries=3, retry_wait=5):
        """
        Convert STITCH CID → PubChem CID → canonical SMILES.

        - Batch requests: up to `batch_size` CIDs per call (far fewer HTTP requests)
        - Correct STITCH parsing: skips flat/stereo indicator digit
        - Exponential backoff retry on HTTP 429/503
        - requests.Session with User-Agent (avoids Colab blocks)
        - Incremental cache save after every batch (safe to interrupt/resume)
        - Single-CID fallback for any batch failures
        - Connectivity pre-check against aspirin (CID 2244)

        Returns dict {STITCH_id: SMILES or None}
        """
        import os, json, time
        try:
            import requests
        except ImportError:
            raise ImportError("pip install requests")

        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        progress_path = cache_path.replace(".csv", "_progress.json")

        # ── helpers ───────────────────────────────────────────────────────────
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})

        def stitch_to_cid(sid):
            # STITCH format: CID + [0|1 flat/stereo] + 8-digit PubChem CID
            # e.g. CID100020430 → skip indicator '1' → CID 20430
            try:
                return int(sid.replace("CID", "")[1:])
            except (ValueError, IndexError):
                return 0

        def make_request(url):
            for attempt in range(max_retries + 1):
                try:
                    r = session.get(url, timeout=15)
                    if r.status_code == 200:
                        return r.content
                    if r.status_code in (429, 503) and attempt < max_retries:
                        wait = retry_wait * (2 ** attempt)
                        print(f"    Rate limit (HTTP {r.status_code}) — "
                              f"retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        return None
                except Exception:
                    if attempt < max_retries:
                        time.sleep(retry_wait)
                    else:
                        return None
            return None

        _SMILES_KEYS = ("CanonicalSMILES", "ConnectivitySMILES", "IsomericSMILES", "SMILES")

        def extract_smiles(prop):
            for key in _SMILES_KEYS:
                if key in prop:
                    return prop[key]
            return None

        def fetch_batch(cids):
            url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                   f"{','.join(str(c) for c in cids)}/property/CanonicalSMILES/JSON")
            raw = make_request(url)
            if not raw:
                return {}
            try:
                props = json.loads(raw).get("PropertyTable", {}).get("Properties", [])
                return {int(p["CID"]): s for p in props if (s := extract_smiles(p))}
            except Exception:
                return {}

        def fetch_single(cid):
            url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
                   f"{cid}/property/CanonicalSMILES/JSON")
            raw = make_request(url)
            if not raw:
                return None
            try:
                props = json.loads(raw).get("PropertyTable", {}).get("Properties", [])
                return extract_smiles(props[0]) if props else None
            except Exception:
                return None

        def save_cache(smiles_dict):
            rows = [{"STITCH": sid, "SMILES": smi}
                    for sid, smi in smiles_dict.items()]
            new_df = pd.DataFrame(rows)
            if os.path.exists(cache_path):
                old_df = pd.read_csv(cache_path)
                combined = pd.concat([old_df, new_df]).drop_duplicates("STITCH")
            else:
                combined = new_df
            combined.to_csv(cache_path, index=False)

        # ── load existing cache ────────────────────────────────────────────────
        smiles = {}
        if os.path.exists(progress_path):
            with open(progress_path) as f:
                smiles = json.load(f)
            print(f"Resuming: {len(smiles)} already fetched.")
        elif os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            smiles = {r["STITCH"]: r["SMILES"]
                      for _, r in df.iterrows() if pd.notna(r["SMILES"])}
            print(f"Loaded {len(smiles)} from existing cache.")

        remaining = [sid for sid in stitch_ids if sid not in smiles]
        print(f"Total: {len(stitch_ids)} | Already cached: {len(smiles)} | "
              f"To fetch: {len(remaining)}")

        if not remaining:
            print("All drugs already cached — nothing to fetch.")
            return {sid: smiles.get(sid) for sid in stitch_ids}

        # ── connectivity check ────────────────────────────────────────────────
        print("Testing PubChem connectivity (aspirin CID 2244)...")
        test = fetch_single(2244)
        if test:
            print(f"  OK — {test[:60]}")
        else:
            print("  FAILED — PubChem not reachable from this environment.")
            print("  Try: !curl -s 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
                  "compound/cid/2244/property/CanonicalSMILES/JSON'")
            return {sid: smiles.get(sid) for sid in stitch_ids}

        # ── batch fetch ───────────────────────────────────────────────────────
        cid_to_stitch = {stitch_to_cid(sid): sid
                         for sid in remaining if stitch_to_cid(sid) > 0}
        valid_cids    = sorted(cid_to_stitch)
        failed_cids   = []

        print(f"\nFetching {len(valid_cids)} drugs in batches of {batch_size}...")
        for i in range(0, len(valid_cids), batch_size):
            batch        = valid_cids[i : i + batch_size]
            batch_result = fetch_batch(batch)

            for cid in batch:
                stitch = cid_to_stitch[cid]
                if cid in batch_result:
                    smiles[stitch] = batch_result[cid]
                else:
                    failed_cids.append(cid)

            done = min(i + batch_size, len(valid_cids))
            print(f"  {done}/{len(valid_cids)}  "
                  f"fetched={len(smiles)}  failed={len(failed_cids)}")

            with open(progress_path, "w") as f:
                json.dump(smiles, f)

            time.sleep(sleep_batch)

        # ── single-CID fallback ───────────────────────────────────────────────
        if failed_cids:
            print(f"\nFallback: retrying {len(failed_cids)} failed CIDs one by one...")
            still_failed = []
            for j, cid in enumerate(failed_cids):
                stitch = cid_to_stitch[cid]
                result = fetch_single(cid)
                if result:
                    smiles[stitch] = result
                else:
                    still_failed.append(stitch)
                if (j + 1) % 20 == 0 or (j + 1) == len(failed_cids):
                    print(f"  {j+1}/{len(failed_cids)}  "
                          f"recovered={len(failed_cids)-len(still_failed)}")
                time.sleep(sleep_single)

            if still_failed:
                print(f"  Could not fetch {len(still_failed)} drugs "
                      f"(will be zero-vectors).")

        # ── final save ────────────────────────────────────────────────────────
        save_cache(smiles)
        if os.path.exists(progress_path):
            os.remove(progress_path)

        n_ok = sum(1 for sid in stitch_ids if smiles.get(sid))
        print(f"\nDone: {n_ok}/{len(stitch_ids)} drugs have SMILES "
              f"({100*n_ok/max(len(stitch_ids),1):.1f}% coverage)")

        return {sid: smiles.get(sid) for sid in stitch_ids}

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
