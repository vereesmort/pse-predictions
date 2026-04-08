"""
02_splitting.py  /  preprocessing/splitting.py
===============================================
Three train/val/test splitting strategies for polypharmacy side effect prediction.

Strategies
----------
1. random_pair       — random split at the (drug_i, drug_j, SE) triple level.
                       Matches prior literature (Decagon, etc.).
                       WARNING: inflates metrics — drugs appear in train & test.

2. stratified_pair   — stratified split preserving positive/negative ratio
                       per side effect in each fold.

3. drug_cold_start   — a held-out subset of drugs is ENTIRELY excluded from
                       training. Simulates prediction on novel drugs.
                       Recommended as the honest evaluation protocol.

All strategies return a SplitResult with train/val/test DataFrames, each
containing columns:
    drug_1, drug_2, side_effect, label  (1=positive, 0=negative)

Usage
-----
    from preprocessing.splitting import Splitter
    splitter = Splitter(combo_df, strategy="drug_cold_start", seed=42)
    splits   = splitter.split(se_id)   # per-SE split
    # or
    all_splits = splitter.split_all()  # dict {se_id: SplitResult}
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional

from configs.config import (
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    COLD_START_DRUG_FRAC, RANDOM_SEED,
)


@dataclass
class SplitResult:
    train : pd.DataFrame
    val   : pd.DataFrame
    test  : pd.DataFrame
    strategy : str
    se_id    : str
    meta     : dict   # e.g., held-out drug set for cold-start

    def sizes(self):
        return {
            "train_pos": int((self.train["label"] == 1).sum()),
            "train_neg": int((self.train["label"] == 0).sum()),
            "val_pos"  : int((self.val["label"] == 1).sum()),
            "val_neg"  : int((self.val["label"] == 0).sum()),
            "test_pos" : int((self.test["label"] == 1).sum()),
            "test_neg" : int((self.test["label"] == 0).sum()),
        }


class Splitter:
    """
    Computes train/val/test splits for polypharmacy side effect data.

    Parameters
    ----------
    combo_df  : filtered combo DataFrame (drug pairs × SE, positives only)
    negatives : DataFrame with columns [drug_1, drug_2, side_effect, label=0]
                produced by preprocessing/sampling.py
    strategy  : "random_pair" | "stratified_pair" | "drug_cold_start"
    seed      : random seed
    """

    def __init__(self, combo_df: pd.DataFrame, negatives: pd.DataFrame,
                 strategy: str = "drug_cold_start", seed: int = RANDOM_SEED):
        self.combo_df  = combo_df.copy()
        self.negatives = negatives.copy()
        self.strategy  = strategy
        self.rng       = np.random.default_rng(seed)
        self.seed      = seed

        # Build positive DataFrame with standardised columns
        self.positives = pd.DataFrame({
            "drug_1"     : combo_df["STITCH 1"],
            "drug_2"     : combo_df["STITCH 2"],
            "side_effect": combo_df["Polypharmacy Side Effect"],
            "label"      : 1,
        })

        self.all_drugs = sorted(
            set(self.positives["drug_1"]).union(self.positives["drug_2"])
        )

        # Pre-compute held-out drug set for cold-start (fixed across all SEs)
        n_holdout = max(1, int(len(self.all_drugs) * COLD_START_DRUG_FRAC))
        shuffled  = self.rng.permutation(self.all_drugs)
        self.held_out_drugs = set(shuffled[:n_holdout].tolist())
        print(f"[Splitter] strategy={strategy}, held-out drugs (cold-start)="
              f"{len(self.held_out_drugs)}/{len(self.all_drugs)}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def split(self, se_id: str) -> SplitResult:
        """Split a single side effect."""
        pos = self.positives[self.positives["side_effect"] == se_id].copy()
        neg = self.negatives[self.negatives["side_effect"] == se_id].copy()
        combined = pd.concat([pos, neg], ignore_index=True)

        if self.strategy == "random_pair":
            return self._random_pair(combined, se_id)
        elif self.strategy == "stratified_pair":
            return self._stratified_pair(combined, se_id)
        elif self.strategy == "drug_cold_start":
            return self._drug_cold_start(pos, neg, se_id)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def split_all(self, se_ids=None) -> Dict[str, SplitResult]:
        """Split all (or specified) side effects."""
        if se_ids is None:
            se_ids = self.positives["side_effect"].unique()
        results = {}
        for se in se_ids:
            results[se] = self.split(se)
        return results

    # ── Strategy implementations ───────────────────────────────────────────────

    def _random_pair(self, combined: pd.DataFrame, se_id: str) -> SplitResult:
        """Randomly shuffle and split triples."""
        idx = self.rng.permutation(len(combined))
        combined = combined.iloc[idx].reset_index(drop=True)

        n = len(combined)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        train = combined.iloc[:n_train]
        val   = combined.iloc[n_train:n_train + n_val]
        test  = combined.iloc[n_train + n_val:]

        return SplitResult(train=train.reset_index(drop=True),
                           val=val.reset_index(drop=True),
                           test=test.reset_index(drop=True),
                           strategy="random_pair", se_id=se_id, meta={})

    def _stratified_pair(self, combined: pd.DataFrame, se_id: str) -> SplitResult:
        """Stratified split preserving pos/neg ratio."""
        from sklearn.model_selection import train_test_split

        pos = combined[combined["label"] == 1]
        neg = combined[combined["label"] == 0]

        def _split_df(df):
            if len(df) < 3:
                return df, df.iloc[:0], df.iloc[:0]
            tr, tmp = train_test_split(df, test_size=(VAL_RATIO + TEST_RATIO),
                                       random_state=self.seed)
            vl, te  = train_test_split(tmp, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
                                       random_state=self.seed)
            return tr, vl, te

        p_tr, p_vl, p_te = _split_df(pos)
        n_tr, n_vl, n_te = _split_df(neg)

        train = pd.concat([p_tr, n_tr], ignore_index=True)
        val   = pd.concat([p_vl, n_vl], ignore_index=True)
        test  = pd.concat([p_te, n_te], ignore_index=True)

        return SplitResult(train=train, val=val, test=test,
                           strategy="stratified_pair", se_id=se_id, meta={})

    def _drug_cold_start(self, pos: pd.DataFrame, neg: pd.DataFrame,
                         se_id: str) -> SplitResult:
        """
        Held-out drug split.

        Pairs where EITHER drug is in the held-out set go to test.
        The remainder is split randomly into train/val.

        This prevents any leakage of a drug's identity between train and test.
        """
        held = self.held_out_drugs

        def _in_holdout(df):
            mask = df["drug_1"].isin(held) | df["drug_2"].isin(held)
            return df[mask], df[~mask]

        pos_test,  pos_train_val = _in_holdout(pos)
        neg_test,  neg_train_val = _in_holdout(neg)

        # Split train_val into train / val (no drug leakage — held-out already removed)
        train_val = pd.concat([pos_train_val, neg_train_val], ignore_index=True)
        idx       = self.rng.permutation(len(train_val))
        train_val = train_val.iloc[idx].reset_index(drop=True)

        n_train = int(len(train_val) * TRAIN_RATIO / (TRAIN_RATIO + VAL_RATIO))
        train   = train_val.iloc[:n_train]
        val     = train_val.iloc[n_train:]

        test = pd.concat([pos_test, neg_test], ignore_index=True)
        test = test.iloc[self.rng.permutation(len(test))].reset_index(drop=True)

        return SplitResult(
            train=train.reset_index(drop=True),
            val=val.reset_index(drop=True),
            test=test.reset_index(drop=True),
            strategy="drug_cold_start", se_id=se_id,
            meta={"held_out_drugs": held},
        )


# ── Comparison utility ─────────────────────────────────────────────────────────

def compare_splits(combo_df, negatives, se_id, seed=RANDOM_SEED):
    """
    Run all three splitting strategies for one SE and print a comparison table.
    Useful for demonstrating the inflation caused by random splitting.
    """
    rows = []
    for strategy in ["random_pair", "stratified_pair", "drug_cold_start"]:
        sp = Splitter(combo_df, negatives, strategy=strategy, seed=seed)
        result = sp.split(se_id)
        sz = result.sizes()
        rows.append({
            "strategy"   : strategy,
            "train_pos"  : sz["train_pos"],
            "train_neg"  : sz["train_neg"],
            "val_pos"    : sz["val_pos"],
            "test_pos"   : sz["test_pos"],
            "test_neg"   : sz["test_neg"],
            # Drug overlap: fraction of test drugs seen in training
            "test_drug_in_train_%": _drug_overlap_pct(result),
        })
    df = pd.DataFrame(rows)
    print(f"\n=== Split comparison for SE {se_id} ===")
    print(df.to_string(index=False))
    return df


def _drug_overlap_pct(result: SplitResult) -> float:
    """Fraction of unique test-set drugs that also appear in the training set."""
    train_drugs = (set(result.train["drug_1"]) | set(result.train["drug_2"]))
    test_drugs  = (set(result.test["drug_1"])  | set(result.test["drug_2"]))
    if not test_drugs:
        return 0.0
    overlap = test_drugs & train_drugs
    return round(100 * len(overlap) / len(test_drugs), 1)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from configs.config import RAW_FILES, MIN_PAIRS_PER_SE, METRICS_DIR, REPRESENTATIVE_SES

    os.makedirs(METRICS_DIR, exist_ok=True)

    combo   = pd.read_csv(RAW_FILES["combo"])
    se_cts  = combo.groupby("Polypharmacy Side Effect").size()
    se_500  = se_cts[se_cts >= MIN_PAIRS_PER_SE].index
    combo_f = combo[combo["Polypharmacy Side Effect"].isin(se_500)].copy()

    # Load negatives produced by sampling.py
    neg_path = os.path.join(METRICS_DIR, "negatives_random.csv")
    if not os.path.exists(neg_path):
        print(f"Negatives file not found at {neg_path}. Run 03_sampling.py first.")
        sys.exit(1)
    negatives = pd.read_csv(neg_path)

    print("Running split comparisons on representative SEs...")
    all_rows = []
    for se_id in list(REPRESENTATIVE_SES)[:3]:  # demo on first 3
        df = compare_splits(combo_f, negatives, se_id)
        df["se_id"] = se_id
        all_rows.append(df)

    summary = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(METRICS_DIR, "split_comparison.csv")
    summary.to_csv(out_path, index=False)
    print(f"\nSaved split comparison to {out_path}")
