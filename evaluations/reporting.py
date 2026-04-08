"""
evaluation/reporting.py
=======================
Visualisation and report generation for experiment results.

Produces:
  - ROC / PR curves per model and representative SE
  - Heatmap: AUROC by model × (size_bin, missing_bin) stratum
  - Bar charts: ablation across feature modes
  - Splitting strategy inflation chart
  - Full LaTeX-ready summary table

Usage
-----
    from evaluation.reporting import Reporter
    reporter = Reporter(aggregator, output_dir="results")
    reporter.plot_all()
    reporter.save_tables()
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, precision_recall_curve

from configs.config import REPRESENTATIVE_SES, PLOTS_DIR, METRICS_DIR


class Reporter:

    def __init__(self, aggregator, output_dir=None):
        self.agg    = aggregator
        self.outdir = output_dir or PLOTS_DIR
        os.makedirs(self.outdir, exist_ok=True)
        os.makedirs(METRICS_DIR, exist_ok=True)

    # ── 1. Summary bar chart: AUROC by model and feature mode ─────────────────

    def plot_model_comparison(self, metric="auroc"):
        df = self.agg.summary()
        if df.empty:
            return
        col = f"{metric}_mean"

        fig, ax = plt.subplots(figsize=(12, 5))
        models  = df["model"].unique()
        modes   = df["feature_mode"].unique()
        x       = np.arange(len(models))
        width   = 0.8 / len(modes)
        colors  = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD"]

        for i, mode in enumerate(modes):
            sub = df[df["feature_mode"] == mode].set_index("model")
            vals = [sub.loc[m, col] if m in sub.index else 0 for m in models]
            bars = ax.bar(x + i * width - 0.4 + width/2, vals, width,
                          label=mode, color=colors[i % len(colors)], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Model comparison — {metric.upper()} by feature mode")
        ax.legend(title="Feature mode", fontsize=9)
        ax.set_ylim(0.5, 1.0)
        plt.tight_layout()
        path = os.path.join(self.outdir, f"fig_model_comparison_{metric}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    # ── 2. Heatmap: AUROC by stratum ──────────────────────────────────────────

    def plot_stratum_heatmap(self, model_name="lightgbm", metric="auroc"):
        df = self.agg.by_stratum()
        if df.empty:
            return
        sub = df[df["model"] == model_name]

        size_order    = ["high", "medium", "low", "very_low"]
        missing_order = ["high", "medium", "low", "very_low"]

        matrix = pd.pivot_table(sub, values=metric,
                                 index="size_bin", columns="missing_bin",
                                 aggfunc="mean")
        matrix = matrix.reindex(index=size_order, columns=missing_order)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(matrix.values, cmap="RdYlGn", vmin=0.5, vmax=1.0,
                       aspect="auto")
        plt.colorbar(im, ax=ax, label=metric.upper())

        ax.set_xticks(range(len(missing_order)))
        ax.set_xticklabels([f"{m}\nmissing" for m in missing_order], fontsize=9)
        ax.set_yticks(range(len(size_order)))
        ax.set_yticklabels([f"{s}\npairs" for s in size_order], fontsize=9)
        ax.set_title(f"{model_name} — {metric.upper()} by stratum")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=8, color="black")

        plt.tight_layout()
        path = os.path.join(self.outdir, f"fig_heatmap_{model_name}_{metric}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    # ── 3. Ablation: feature mode comparison ──────────────────────────────────

    def plot_ablation(self, strategy="drug_cold_start"):
        df = self.agg.ablation_table()
        if df.empty:
            return
        sub = df[df["strategy"] == strategy]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        colors = {"target_only": "#378ADD", "fp_only": "#D85A30",
                  "target+fp": "#1D9E75", "all": "#7F77DD"}

        for ax, metric in zip(axes, ["auroc", "ap"]):
            models = sub["model"].unique()
            modes  = sub["feature_mode"].unique()
            x      = np.arange(len(models))
            w      = 0.7 / len(modes)
            for i, mode in enumerate(modes):
                vals = []
                for m in models:
                    row = sub[(sub["model"] == m) & (sub["feature_mode"] == mode)]
                    vals.append(row[metric].values[0] if len(row) else 0)
                ax.bar(x + i * w, vals, w,
                       label=mode, color=colors.get(mode, "#888"),
                       alpha=0.85)
            ax.set_xticks(x + w * (len(modes)-1) / 2)
            ax.set_xticklabels(models, rotation=15)
            ax.set_ylabel(metric.upper())
            ax.set_title(f"Feature mode ablation — {metric.upper()}\n({strategy})")
            ax.set_ylim(0.45, 1.0)
            ax.legend(title="Feature mode", fontsize=8)

        plt.tight_layout()
        path = os.path.join(self.outdir, f"fig_ablation_{strategy}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    # ── 4. Split inflation chart ───────────────────────────────────────────────

    def plot_split_inflation(self, split_comparison_df: pd.DataFrame):
        """
        Show how reported AUROC inflates under random vs cold-start splitting.

        split_comparison_df : output of compare_splits() from preprocessing/splitting.py
                              should have columns: strategy, se_id, auroc
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        strategies = split_comparison_df["strategy"].unique()
        colors_map = {"random_pair": "#E24B4A", "stratified_pair": "#EF9F27",
                      "drug_cold_start": "#1D9E75"}

        for strat in strategies:
            sub = split_comparison_df[split_comparison_df["strategy"] == strat]
            sub = sub.sort_values("se_id")
            ax.plot(range(len(sub)), sub["auroc"],
                    label=strat, color=colors_map.get(strat, "#888"),
                    alpha=0.8, linewidth=1.5)

        ax.set_xlabel("Side effect (sorted by SE id)")
        ax.set_ylabel("AUROC")
        ax.set_title("AUROC inflation: random vs. drug cold-start splitting")
        ax.legend()
        ax.set_ylim(0.4, 1.0)
        plt.tight_layout()
        path = os.path.join(self.outdir, "fig_split_inflation.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    # ── 5. ROC / PR curves for representative SEs ─────────────────────────────

    def plot_roc_pr(self, detailed_results: pd.DataFrame, se_id: str):
        """
        Plot ROC and PR curves for all models on a specific SE.

        detailed_results : must have columns [se_id, model, y_true, y_score]
        """
        sub = detailed_results[detailed_results["se_id"] == se_id]
        if sub.empty:
            return

        se_name = REPRESENTATIVE_SES.get(se_id, se_id)
        colors  = ["#378ADD", "#1D9E75", "#D85A30", "#7F77DD", "#D4537E"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, curve_type in zip(axes, ["roc", "pr"]):
            for i, (_, row) in enumerate(sub.iterrows()):
                y_true  = np.array(row["y_true"])
                y_score = np.array(row["y_score"])
                color   = colors[i % len(colors)]
                label   = f"{row['model']} ({row['feature_mode']})"

                if curve_type == "roc":
                    fpr, tpr, _ = roc_curve(y_true, y_score)
                    ax.plot(fpr, tpr, color=color, alpha=0.8, label=label)
                    ax.plot([0,1], [0,1], "k--", linewidth=0.8)
                    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
                    ax.set_title(f"ROC — {se_name}")
                else:
                    prec, rec, _ = precision_recall_curve(y_true, y_score)
                    ax.plot(rec, prec, color=color, alpha=0.8, label=label)
                    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
                    ax.set_title(f"PR — {se_name}")

            ax.legend(fontsize=7)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        plt.tight_layout()
        safe_name = se_id.replace("/", "_")
        path = os.path.join(self.outdir, f"fig_roc_pr_{safe_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    # ── 6. Missing-target regime ablation ─────────────────────────────────────

    def plot_missing_target_ablation(self):
        df = self.agg.missing_target_ablation()
        if df.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        regimes   = ["both_have", "one_has", "neither_has"]
        colors    = {"both_have": "#1D9E75", "one_has": "#EF9F27",
                     "neither_has": "#E24B4A"}

        for ax, metric in zip(axes, ["auroc", "ap"]):
            models = df["model"].unique()
            x      = np.arange(len(models))
            w      = 0.7 / len(regimes)
            for i, reg in enumerate(regimes):
                vals = []
                for m in models:
                    row = df[(df["model"] == m) & (df["target_regime"] == reg)]
                    vals.append(row[metric].values[0] if len(row) else 0)
                ax.bar(x + i * w, vals, w, label=reg,
                       color=colors[reg], alpha=0.85)
            ax.set_xticks(x + w)
            ax.set_xticklabels(models, rotation=15)
            ax.set_ylabel(metric.upper())
            ax.set_title(f"Target regime ablation — {metric.upper()}")
            ax.legend(title="Drug pair target coverage", fontsize=8)
            ax.set_ylim(0.4, 1.0)

        plt.tight_layout()
        path = os.path.join(self.outdir, "fig_target_regime_ablation.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")

    # ── 7. Save all tables ─────────────────────────────────────────────────────

    def save_tables(self):
        all_res   = self.agg.all_results()
        summary   = self.agg.summary()
        stratum   = self.agg.by_stratum()
        ablation  = self.agg.ablation_table()
        rep_table = self.agg.representative_se_table(REPRESENTATIVE_SES)

        for df, name in [
            (all_res,   "all_results"),
            (summary,   "summary"),
            (stratum,   "by_stratum"),
            (ablation,  "ablation"),
            (rep_table, "representative_ses"),
        ]:
            if df is not None and len(df) > 0:
                path = os.path.join(METRICS_DIR, f"{name}.csv")
                df.to_csv(path, index=False)
                print(f"Saved {path}")

    # ── Convenience ───────────────────────────────────────────────────────────

    def plot_all(self):
        print("\n=== Generating all plots ===")
        for model in ["lightgbm", "xgboost", "catboost", "rgcn"]:
            self.plot_stratum_heatmap(model_name=model)
        self.plot_model_comparison(metric="auroc")
        self.plot_model_comparison(metric="ap")
        for strategy in ["random_pair", "drug_cold_start"]:
            self.plot_ablation(strategy=strategy)
        self.plot_missing_target_ablation()
        print("=== Plots done ===\n")
