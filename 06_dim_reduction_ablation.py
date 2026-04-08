"""
06_dim_reduction_ablation.py
============================
Ablation study comparing all dimensionality reduction methods as drug
feature extractors for polypharmacy side effect prediction.

Experimental design
-------------------
  Fixed:  LightGBM, drug_cold_start split, structured negatives,
          hadamard_absdiff pair operator, 12 representative SEs
  Varied: dim reduction method (PCA, KernelPCA, GraphDiffusedSVD,
          Node2Vec, GraphAutoencoder, UMAP)

For each method, two feature modes are evaluated:
  - dim_only      : reduced drug embedding only
  - dim+fp        : reduced embedding concatenated with Morgan fingerprints

This isolates the contribution of the reduction method from the
fingerprint fallback for no-target drugs.

Outputs
-------
  results/metrics/dim_reduction_comparison.csv
  results/metrics/dim_reduction_summary.csv
  results/plots/fig_dim_reduction_*.png

Run:
    python 06_dim_reduction_ablation.py [--fast]   (--fast skips Node2Vec/GAE)
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from configs.config import (
    RAW_FILES, METRICS_DIR, PLOTS_DIR, CACHE_DIR,
    MIN_PAIRS_PER_SE, REPRESENTATIVE_SES, RANDOM_SEED, NEG_RATIO,
)
from features.dim_reduction import DrugEmbeddingPipeline
from features.drug_features import DrugFeatureBuilder
from features.pair_operators import make_pair_features
from preprocessing.sampling import NegativeSampler
from preprocessing.splitting import Splitter
from models.lightgbm_model import LGBMPredictor
from evaluation.metrics import compute_metrics

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs(CACHE_DIR,   exist_ok=True)

METHOD_LABELS = {
    "pca"               : "PCA\n(baseline)",
    "kernel_pca"        : "Kernel PCA\n(RBF)",
    "graph_diffused_svd": "Graph-Diffused\nSVD (ours)",
    "node2vec"          : "Node2Vec\naggregation",
    "graph_autoencoder" : "Graph\nAutoencoder",
    "umap"              : "UMAP\n(diffused)",
}
METHOD_COLORS = {
    "pca"               : "#B4B2A9",   # gray — baseline
    "kernel_pca"        : "#85B7EB",   # light blue
    "graph_diffused_svd": "#1D9E75",   # teal — primary
    "node2vec"          : "#7F77DD",   # purple
    "graph_autoencoder" : "#D85A30",   # coral — strongest
    "umap"              : "#EF9F27",   # amber
}


def load_and_prepare(se_ids, neg_strategy="structured"):
    print("Loading data...")
    combo   = pd.read_csv(RAW_FILES["combo"])
    targets = pd.read_csv(RAW_FILES["targets"])
    ppi     = pd.read_csv(RAW_FILES["ppi"])
    se_cts  = combo.groupby("Polypharmacy Side Effect").size()
    se_500  = se_cts[se_cts >= MIN_PAIRS_PER_SE].index
    combo_f = combo[combo["Polypharmacy Side Effect"].isin(se_500)].copy()
    all_drugs = sorted(pd.unique(combo_f[["STITCH 1","STITCH 2"]].values.ravel()))

    # Negatives
    neg_cache = os.path.join(CACHE_DIR, f"negatives_{neg_strategy}_rep.csv")
    if os.path.exists(neg_cache):
        negatives = pd.read_csv(neg_cache)
    else:
        sampler   = NegativeSampler(combo_f, strategy=neg_strategy, seed=RANDOM_SEED)
        negatives = sampler.sample_all(neg_ratio=NEG_RATIO, se_ids=se_ids)
        negatives.to_csv(neg_cache, index=False)

    # Splits
    splitter = Splitter(combo_f, negatives, strategy="drug_cold_start",
                        seed=RANDOM_SEED)
    splits   = {se: splitter.split(se) for se in se_ids}

    # Morgan fingerprints (for +fp mode)
    smiles_cache = os.path.join(CACHE_DIR, "smiles.csv")
    smiles_dict  = {}
    if os.path.exists(smiles_cache):
        sc = pd.read_csv(smiles_cache)
        smiles_dict = {r["STITCH"]: r["SMILES"] for _, r in sc.iterrows()
                       if pd.notna(r["SMILES"])}
        print(f"  Loaded {len(smiles_dict)} SMILES from cache")

    # Build Morgan fingerprints as numpy array
    builder = DrugFeatureBuilder(targets, ppi)
    fp_mat, has_smiles = builder.build_morgan_fingerprints(all_drugs, smiles_dict)
    t_drugs = set(targets["STITCH"])
    has_target = np.array([d in t_drugs for d in all_drugs])

    return combo_f, targets, ppi, all_drugs, splits, smiles_dict, fp_mat, has_target


class EmbeddingFeatureSet:
    """
    Wraps a (n_drugs, embed_dim) embedding matrix with the same .get() API
    as DrugFeatureSet, so it plugs directly into LGBMPredictor.
    Optionally appends Morgan fingerprints.
    """
    def __init__(self, drugs, Z, fp_mat=None, mode="dim_only"):
        self._drug2idx = {d: i for i, d in enumerate(drugs)}
        self._Z        = Z          # (n_drugs, embed_dim)
        self._fp       = fp_mat     # (n_drugs, fp_dim) or None
        self._mode     = mode

    def get(self, drug_ids, mode=None):
        idx = np.array([self._drug2idx[d] for d in drug_ids])
        Z   = self._Z[idx]
        if self._fp is not None and self._mode == "dim+fp":
            return np.concatenate([Z, self._fp[idx]], axis=1)
        return Z

    def feature_dim(self, mode=None):
        base = self._Z.shape[1]
        if self._fp is not None and self._mode == "dim+fp":
            return base + self._fp.shape[1]
        return base


def evaluate_embedding(Z, fp_mat, drugs, splits, se_ids, feature_mode,
                       method_name, verbose=False):
    """Run LightGBM on one embedding matrix and return per-SE results."""
    fset = EmbeddingFeatureSet(drugs, Z, fp_mat, mode=feature_mode)
    lgbm = LGBMPredictor(fset, operator="hadamard_absdiff",
                         feature_mode="all")   # 'all' is ignored — fset.get handles it
    # Monkey-patch fset.get to always return correct mode
    rows = []
    for se_id in se_ids:
        split = splits[se_id]
        if len(split.train) == 0 or split.train["label"].nunique() < 2:
            continue
        lgbm.fit_one(se_id, split.train, split.val)
        proba   = lgbm.predict_proba(se_id, split.test)
        y_true  = split.test["label"].values
        metrics = compute_metrics(y_true, proba)
        rows.append({"se_id": se_id, "method": method_name,
                     "feature_mode": feature_mode,
                     "model": "lightgbm", **metrics})
        if verbose:
            print(f"    {se_id}: AUROC={metrics['auroc']:.3f} AP={metrics['ap']:.3f}")
    return pd.DataFrame(rows)


def run_dim_reduction_ablation(fast=False):
    se_ids = list(REPRESENTATIVE_SES.keys())

    (combo_f, targets, ppi, all_drugs, splits,
     smiles_dict, fp_mat, has_target) = load_and_prepare(se_ids)

    # Choose which methods to run
    if fast:
        methods = ["pca", "kernel_pca", "graph_diffused_svd", "umap"]
        print("[fast mode] Skipping Node2Vec and GraphAutoencoder")
    else:
        methods = DrugEmbeddingPipeline.METHODS

    # Build all embeddings
    pipeline = DrugEmbeddingPipeline(targets, ppi, smiles_dict,
                                     random_state=RANDOM_SEED)
    print("\n=== Building drug embeddings ===")
    emb_results = pipeline.build_all(
        all_drugs, n_components=64, methods=methods,
        node2vec_walks=50 if fast else 200,
        gae_epochs=40 if fast else 100,
    )

    # Summary table
    summary_tbl = pipeline.summary_table(emb_results)
    print("\n=== Embedding method summary ===")
    print(summary_tbl.to_string(index=False))
    summary_tbl.to_csv(os.path.join(METRICS_DIR, "dim_reduction_embedding_summary.csv"),
                       index=False)

    # Evaluate each embedding on downstream prediction
    print("\n=== Evaluating on polypharmacy prediction ===")
    all_rows = []
    for method_name, emb_res in emb_results.items():
        Z = emb_res.embeddings
        for feat_mode in ["dim_only", "dim+fp"]:
            if feat_mode == "dim+fp" and fp_mat is None:
                continue
            print(f"\n  {method_name} | {feat_mode}")
            df = evaluate_embedding(Z, fp_mat, all_drugs, splits, se_ids,
                                    feat_mode, method_name, verbose=True)
            all_rows.append(df)

    results = pd.concat(all_rows, ignore_index=True)

    # Add SE metadata
    meta_path = os.path.join(METRICS_DIR, "per_se_missing_targets.csv")
    if os.path.exists(meta_path):
        meta = pd.read_csv(meta_path)
        results = results.merge(
            meta[["Polypharmacy Side Effect","Total Pairs",
                  "Pct Drugs Without Target"]],
            left_on="se_id", right_on="Polypharmacy Side Effect", how="left"
        ).drop(columns=["Polypharmacy Side Effect"])

    results.to_csv(os.path.join(METRICS_DIR, "dim_reduction_comparison.csv"),
                   index=False)
    print(f"\nSaved dim_reduction_comparison.csv ({len(results)} rows)")

    # Summary: mean AUROC/AP per method
    summary = (results.groupby(["method","feature_mode"])[["auroc","ap"]]
                      .mean().round(4).reset_index()
                      .sort_values(["feature_mode","auroc"], ascending=[True, False]))
    summary.to_csv(os.path.join(METRICS_DIR, "dim_reduction_summary.csv"), index=False)
    print("\n=== Summary (mean across 12 SEs) ===")
    print(summary.to_string(index=False))

    # Generate plots
    plot_method_comparison(results, methods)
    plot_missing_target_breakdown(results, methods, has_target, all_drugs, splits)
    plot_embedding_visualisation(emb_results, has_target)

    return results


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_method_comparison(results, methods):
    """Bar chart: AUROC and AP per reduction method, split by feature mode."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    feat_modes  = ["dim_only", "dim+fp"]
    bar_alpha   = [0.55, 0.95]   # dim_only lighter, dim+fp solid
    x           = np.arange(len(methods))
    w           = 0.35

    for ax, metric in zip(axes, ["auroc", "ap"]):
        for j, fm in enumerate(feat_modes):
            sub  = results[results["feature_mode"] == fm]
            mean = sub.groupby("method")[metric].mean()
            std  = sub.groupby("method")[metric].std().fillna(0)
            vals = [float(mean.get(m, 0)) for m in methods]
            errs = [float(std.get(m, 0)) for m in methods]
            bars = ax.bar(x + j*w - w/2, vals, w,
                          color=[METHOD_COLORS.get(m,"#888") for m in methods],
                          alpha=bar_alpha[j],
                          label=fm, yerr=errs, capsize=3,
                          error_kw={"linewidth": 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods],
                           fontsize=8.5)
        ax.set_ylabel(metric.upper(), fontsize=10)
        ax.set_ylim(max(0, ax.get_ylim()[0] - 0.02), min(1, ax.get_ylim()[1] + 0.03))
        ax.axhline(0, color="black", linewidth=0.4)
        ax.set_title(f"Dim reduction comparison — {metric.upper()}", fontsize=11)
        ax.legend(title="Feature mode", fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_dim_reduction_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_missing_target_breakdown(results, methods, has_target, drugs, splits):
    """
    Show AUROC per method broken down by % of drugs missing targets per SE.
    Highlights where graph-aware methods help most.
    """
    meta_path = os.path.join(METRICS_DIR, "per_se_missing_targets.csv")
    if not os.path.exists(meta_path):
        return
    meta = pd.read_csv(meta_path).set_index("Polypharmacy Side Effect")

    # Bin SEs by missing target %
    def miss_bin(pct):
        if pct > 55: return "high (>55%)"
        if pct > 41: return "medium (41-55%)"
        return "low (<41%)"

    sub = results[results["feature_mode"] == "dim+fp"].copy()
    sub["miss_bin"] = sub["se_id"].map(
        lambda s: miss_bin(meta.loc[s, "Pct Drugs Without Target"])
        if s in meta.index else "unknown"
    )
    sub = sub[sub["miss_bin"] != "unknown"]

    bins     = ["low (<41%)", "medium (41-55%)", "high (>55%)"]
    fig, axes = plt.subplots(1, len(bins), figsize=(14, 5), sharey=True)

    for ax, b in zip(axes, bins):
        grp  = sub[sub["miss_bin"] == b]
        mean = grp.groupby("method")["auroc"].mean()
        vals = [float(mean.get(m, 0)) for m in methods]
        bars = ax.barh([METHOD_LABELS.get(m, m) for m in methods], vals,
                       color=[METHOD_COLORS.get(m,"#888") for m in methods],
                       alpha=0.85)
        ax.set_xlabel("AUROC")
        ax.set_title(f"Missing targets: {b}", fontsize=9)
        ax.set_xlim(0.4, 1.0)
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                        f"{val:.3f}", va="center", fontsize=7.5)

    plt.suptitle("AUROC by dim-reduction method × missing target rate",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_dim_reduction_missing_target.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_embedding_visualisation(emb_results, has_target, n_components_vis=2):
    """
    2D UMAP projections of each embedding space, coloured by target availability.
    Helps visualise how much each method clusters drugs by biological similarity.
    """
    try:
        import umap as _umap
    except ImportError:
        print("  Skipping embedding visualisation (umap not available)")
        return

    methods_to_plot = [m for m in emb_results if emb_results[m].embeddings.shape[0] > 10]
    ncols = min(3, len(methods_to_plot))
    nrows = (len(methods_to_plot) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.array(axes).flatten()

    for i, method in enumerate(methods_to_plot):
        ax = axes[i]
        Z  = emb_results[method].embeddings

        # Project to 2D if needed
        if Z.shape[1] > 2:
            reducer = _umap.UMAP(n_components=2, random_state=42, n_jobs=1)
            Z2d = reducer.fit_transform(Z)
        else:
            Z2d = Z

        colors = np.where(has_target, "#1D9E75", "#E24B4A")
        ax.scatter(Z2d[~has_target, 0], Z2d[~has_target, 1],
                   c="#E24B4A", s=18, alpha=0.6, label="No target", zorder=2)
        ax.scatter(Z2d[has_target, 0], Z2d[has_target, 1],
                   c="#1D9E75", s=18, alpha=0.6, label="Has target", zorder=3)
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=7, markerscale=1.5)

    # Hide unused axes
    for j in range(len(methods_to_plot), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("2D UMAP projection of drug embeddings\n"
                 "(green = has protein target, red = no protein target)",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "fig_dim_reduction_embedding_space.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Skip Node2Vec and GAE (faster, fewer methods)")
    args = parser.parse_args()
    run_dim_reduction_ablation(fast=args.fast)
