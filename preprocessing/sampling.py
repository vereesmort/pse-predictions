"""
03_sampling.py  /  preprocessing/sampling.py
=============================================
Negative sample generation for polypharmacy side effect prediction.

Two strategies
--------------
1. random      — drug pairs NOT present in the combo dataset for the target SE.
                 Standard approach in prior work. Easy negatives.

2. structured  — drug pairs that DO appear in the combo dataset but for a
                 DIFFERENT side effect. Harder negatives: the pair is known
                 to interact pharmacologically, but not via the target SE.
                 Reduces the risk of models simply learning "is this a known pair".

Both strategies guarantee:
  - No (drug_i, drug_j, SE) triple that exists in the positive set
  - Symmetric: we treat (A,B) and (B,A) as the same pair
  - No self-loops (drug_i == drug_j)

Usage (as module)
-----------------
    from preprocessing.sampling import NegativeSampler
    sampler = NegativeSampler(combo_filtered, strategy="structured", seed=42)
    negatives = sampler.sample_all(neg_ratio=1.0)

Usage (CLI)
-----------
    python 03_sampling.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from typing import Literal

from configs.config import RAW_FILES, MIN_PAIRS_PER_SE, METRICS_DIR, NEG_RATIO, RANDOM_SEED


class NegativeSampler:
    """
    Generates negative drug pair samples for each side effect.

    Parameters
    ----------
    combo_df  : filtered positive combo DataFrame
    strategy  : "random" or "structured"
    seed      : random seed
    """

    def __init__(self, combo_df: pd.DataFrame,
                 strategy: Literal["random", "structured"] = "structured",
                 seed: int = RANDOM_SEED):
        self.strategy = strategy
        self.rng      = np.random.default_rng(seed)

        # Canonical positive set: frozensets for symmetry
        self.combo_df  = combo_df.copy()
        self.all_drugs = sorted(
            set(combo_df["STITCH 1"]).union(set(combo_df["STITCH 2"]))
        )
        self.drug_arr  = np.array(self.all_drugs)

        # Positive set per SE: {se_id: set of frozensets}
        print("  Indexing positive pairs...")
        self.pos_per_se: dict[str, set] = {}
        self.all_pos_pairs: set = set()  # all positive pairs regardless of SE
        for _, row in combo_df.iterrows():
            pair = frozenset([row["STITCH 1"], row["STITCH 2"]])
            se   = row["Polypharmacy Side Effect"]
            if se not in self.pos_per_se:
                self.pos_per_se[se] = set()
            self.pos_per_se[se].add(pair)
            self.all_pos_pairs.add((se, pair))

        # For structured negatives: pair → list of SEs it appears in
        if strategy == "structured":
            print("  Building cross-SE pair index for structured negatives...")
            self.pair_to_ses: dict = {}
            for (se, pair) in self.all_pos_pairs:
                self.pair_to_ses.setdefault(pair, set()).add(se)

        print(f"[NegativeSampler] strategy={strategy}, drugs={len(self.all_drugs)}, "
              f"SEs={len(self.pos_per_se)}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def sample(self, se_id: str, neg_ratio: float = NEG_RATIO) -> pd.DataFrame:
        """Sample negatives for one side effect."""
        n_pos = len(self.pos_per_se.get(se_id, set()))
        n_neg = max(1, int(n_pos * neg_ratio))

        if self.strategy == "random":
            return self._sample_random(se_id, n_neg)
        elif self.strategy == "structured":
            return self._sample_structured(se_id, n_neg)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def sample_all(self, neg_ratio: float = NEG_RATIO,
                   se_ids=None) -> pd.DataFrame:
        """Sample negatives for all (or specified) SEs. Returns combined DataFrame."""
        if se_ids is None:
            se_ids = list(self.pos_per_se.keys())
        frames = []
        for i, se in enumerate(se_ids):
            if i % 100 == 0:
                print(f"  Sampling SE {i}/{len(se_ids)}...")
            frames.append(self.sample(se, neg_ratio))
        return pd.concat(frames, ignore_index=True)

    # ── Random negatives ───────────────────────────────────────────────────────

    def _sample_random(self, se_id: str, n_neg: int) -> pd.DataFrame:
        """
        Draw random drug pairs not in the positive set for this SE.
        No restriction on whether the pair appears for other SEs.
        """
        pos_pairs = self.pos_per_se.get(se_id, set())
        sampled   = []
        attempts  = 0
        max_attempts = n_neg * 20

        while len(sampled) < n_neg and attempts < max_attempts:
            i, j = self.rng.choice(len(self.drug_arr), size=2, replace=False)
            d1, d2 = self.drug_arr[i], self.drug_arr[j]
            pair = frozenset([d1, d2])
            if pair not in pos_pairs:
                sampled.append((d1, d2))
                pos_pairs.add(pair)   # avoid duplicates within this batch
            attempts += 1

        return self._to_df(sampled, se_id)

    # ── Structured negatives ───────────────────────────────────────────────────

    def _sample_structured(self, se_id: str, n_neg: int) -> pd.DataFrame:
        """
        Draw drug pairs that exist for a DIFFERENT SE but not for se_id.
        Falls back to random negatives if the cross-SE pool is insufficient.
        """
        pos_for_this_se = self.pos_per_se.get(se_id, set())

        # Candidate cross-SE pairs: pairs known for some OTHER SE
        cross_candidates = [
            pair for pair, ses in self.pair_to_ses.items()
            if se_id not in ses and pair not in pos_for_this_se
        ]
        self.rng.shuffle(cross_candidates)

        sampled = []
        for pair in cross_candidates[:n_neg]:
            drugs = list(pair)
            if len(drugs) == 2:
                sampled.append((drugs[0], drugs[1]))

        # If not enough structured negatives, top up with random
        shortfall = n_neg - len(sampled)
        if shortfall > 0:
            extra_df = self._sample_random(se_id, shortfall)
            extra    = list(zip(extra_df["drug_1"], extra_df["drug_2"]))
            sampled.extend(extra)

        return self._to_df(sampled[:n_neg], se_id)

    # ── Helper ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_df(samples: list, se_id: str) -> pd.DataFrame:
        if not samples:
            return pd.DataFrame(columns=["drug_1", "drug_2", "side_effect", "label"])
        d1, d2 = zip(*samples)
        return pd.DataFrame({
            "drug_1"     : list(d1),
            "drug_2"     : list(d2),
            "side_effect": se_id,
            "label"      : 0,
        })


# ── Sampling comparison utility ────────────────────────────────────────────────

def compare_sampling_strategies(combo_df, se_id, seed=RANDOM_SEED):
    """
    Show the difference between random and structured negatives for one SE.
    Structured negatives are harder: every negative pair is a real drug interaction.
    """
    results = {}
    for strat in ["random", "structured"]:
        sampler  = NegativeSampler(combo_df, strategy=strat, seed=seed)
        neg      = sampler.sample(se_id)
        pos_pairs = sampler.pos_per_se  # all SE positive pairs

        # What fraction of structured negatives are real drug pairs (any SE)?
        n_known_pairs = sum(
            1 for _, row in neg.iterrows()
            if frozenset([row["drug_1"], row["drug_2"]]) in sampler.pair_to_ses
        ) if strat == "structured" else 0

        results[strat] = {
            "n_negatives"           : len(neg),
            "pct_known_pair_any_se" : round(100 * n_known_pairs / max(1, len(neg)), 1),
        }
        print(f"  [{strat}] negatives={len(neg)}, "
              f"known_pair_any_SE={results[strat]['pct_known_pair_any_se']}%")
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(METRICS_DIR, exist_ok=True)

    print("Loading combo data...")
    combo   = pd.read_csv(RAW_FILES["combo"])
    se_cts  = combo.groupby("Polypharmacy Side Effect").size()
    se_500  = se_cts[se_cts >= MIN_PAIRS_PER_SE].index
    combo_f = combo[combo["Polypharmacy Side Effect"].isin(se_500)].copy()

    for strategy in ["random", "structured"]:
        print(f"\n=== Generating {strategy} negatives (ratio={NEG_RATIO}) ===")
        sampler  = NegativeSampler(combo_f, strategy=strategy, seed=RANDOM_SEED)
        neg_df   = sampler.sample_all(neg_ratio=NEG_RATIO)
        out_path = os.path.join(METRICS_DIR, f"negatives_{strategy}.csv")
        neg_df.to_csv(out_path, index=False)
        print(f"  Saved {len(neg_df)} negatives → {out_path}")

    # Comparison demo on first representative SE
    from configs.config import REPRESENTATIVE_SES
    demo_se = list(REPRESENTATIVE_SES.keys())[0]
    print(f"\n=== Sampling strategy comparison for {demo_se} ===")
    compare_sampling_strategies(combo_f, demo_se)
