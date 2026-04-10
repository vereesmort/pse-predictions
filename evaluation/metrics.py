"""
evaluation/metrics.py
=====================
Unified evaluation metrics for polypharmacy side effect prediction.

All models produce probability scores in [0,1] per drug pair.
Metrics computed:
  - AUROC       : area under ROC curve
  - AP          : average precision (area under PR curve)
  - F1          : at threshold 0.5
  - Precision   : at threshold 0.5
  - Recall      : at threshold 0.5
  - AUPRC       : alias for AP

Usage
-----
    from evaluation.metrics import compute_metrics, MetricsAggregator
    m = compute_metrics(y_true, y_score)
    # m = {"auroc": 0.82, "ap": 0.76, "f1": 0.71, ...}
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve,
)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """
    Compute standard classification metrics.

    Parameters
    ----------
    y_true   : binary ground truth labels (0/1)
    y_score  : predicted probabilities in [0, 1]
    threshold: decision threshold for F1/P/R

    Returns
    -------
    dict with keys: auroc, ap, f1, precision, recall
    """
    y_pred = (y_score >= threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        # Degenerate case — only one class in test set
        return {"auroc": float("nan"), "ap": float("nan"),
                "f1": float("nan"), "precision": float("nan"),
                "recall": float("nan")}

    return {
        "auroc"    : round(float(roc_auc_score(y_true, y_score)), 4),
        "ap"       : round(float(average_precision_score(y_true, y_score)), 4),
        "f1"       : round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall"   : round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
    }


def optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Find threshold maximising F1 on the given data."""
    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
    f1s = 2 * prec * rec / np.where(prec + rec == 0, 1, prec + rec)
    return float(thresholds[np.argmax(f1s[:-1])])


# ── Aggregator ─────────────────────────────────────────────────────────────────

class MetricsAggregator:
    """
    Collects per-SE results and produces stratified summary tables.

    Usage
    -----
        agg = MetricsAggregator(se_metadata_df)
        agg.add(results_df)          # from any model .run() method
        summary = agg.summary()      # aggregated by model, feature_mode, strategy
        stratified = agg.by_stratum()  # broken down by sample-size × missing-% bins
    """

    def __init__(self, se_meta: pd.DataFrame):
        """
        se_meta : DataFrame with columns:
            Polypharmacy Side Effect, Total Pairs, Pct Drugs Without Target
        """
        self.se_meta = se_meta.copy()
        self._assign_bins()
        self.records = []

    def _assign_bins(self):
        df = self.se_meta
        pairs = df["Total Pairs"]
        miss  = df["Pct Drugs Without Target"]

        # Use data-driven quartile thresholds so bins are always populated,
        # whether running with 12 representative SEs or all 963.
        sq = pairs.quantile([0.25, 0.5, 0.75])
        mq = miss.quantile([0.25, 0.5, 0.75])

        def size_bin(p):
            if p > sq[0.75]: return "high"
            if p > sq[0.50]: return "medium"
            if p > sq[0.25]: return "low"
            return "very_low"

        def miss_bin(pct):
            if pct > mq[0.75]: return "high"
            if pct > mq[0.50]: return "medium"
            if pct > mq[0.25]: return "low"
            return "very_low"

        df["size_bin"]    = pairs.apply(size_bin)
        df["missing_bin"] = miss.apply(miss_bin)

    def add(self, results_df: pd.DataFrame):
        """Add a results DataFrame from a model .run() call."""
        self.records.append(results_df)

    def all_results(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        df = pd.concat(self.records, ignore_index=True)
        # Join stratum metadata
        df = df.merge(
            self.se_meta[["Polypharmacy Side Effect", "size_bin", "missing_bin",
                           "Total Pairs", "Pct Drugs Without Target"]],
            left_on="se_id", right_on="Polypharmacy Side Effect", how="left"
        ).drop(columns=["Polypharmacy Side Effect"])
        return df

    def summary(self) -> pd.DataFrame:
        """Aggregate mean ± std across SEs, grouped by (model, feature_mode, strategy)."""
        df = self.all_results()
        if df.empty:
            return df
        metric_cols = ["auroc", "ap", "f1", "precision", "recall"]
        group_cols  = ["model", "feature_mode", "strategy"]
        agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).round(4)
        agg.columns = ["_".join(c) for c in agg.columns]
        return agg.reset_index()

    def by_stratum(self) -> pd.DataFrame:
        """Break down AUROC and AP by (model, size_bin, missing_bin)."""
        df = self.all_results()
        if df.empty:
            return df
        return (df.groupby(["model", "feature_mode", "size_bin", "missing_bin"])
                  [["auroc", "ap"]].mean().round(4).reset_index())

    def representative_se_table(self, rep_ses: dict) -> pd.DataFrame:
        """
        Return a detailed per-SE table for the 12 representative side effects.

        rep_ses : {se_id: readable_name}
        """
        df = self.all_results()
        if df.empty:
            return df
        rep_df = df[df["se_id"].isin(rep_ses)].copy()
        rep_df["SE Name"] = rep_df["se_id"].map(rep_ses)
        return rep_df.sort_values(["SE Name", "model"])

    def ablation_table(self) -> pd.DataFrame:
        """
        Ablation: compare feature modes (target_only vs fp_only vs target+fp)
        for the same model and strategy.
        """
        df = self.all_results()
        if df.empty:
            return df
        return (df.groupby(["model", "strategy", "feature_mode"])
                  [["auroc", "ap"]].mean().round(4).reset_index()
                  .sort_values(["model", "strategy", "auroc"], ascending=[True, True, False]))

    def missing_target_ablation(self) -> pd.DataFrame:
        """
        Break down performance by per-pair target availability:
          - both drugs have targets
          - one drug has targets
          - neither drug has targets
        Requires 'target_regime' column in results (added by run_with_regime).
        """
        df = self.all_results()
        if "target_regime" not in df.columns:
            print("[MetricsAggregator] 'target_regime' column not found. "
                  "Run evaluate_with_regime() to populate it.")
            return df
        return (df.groupby(["model", "feature_mode", "target_regime"])
                  [["auroc", "ap"]].mean().round(4).reset_index())
