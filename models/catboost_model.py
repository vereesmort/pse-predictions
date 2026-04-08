"""
models/catboost_model.py
========================
CatBoost-based polypharmacy side effect predictor.

Two modes:
  per_se     — one model per side effect (same as LGBM/XGB)
  multitask  — single model with side_effect_id as categorical feature.
               This is unique to CatBoost's native categorical handling.

Usage
-----
    from models.catboost_model import CatBoostPredictor
    model = CatBoostPredictor(feature_set, mode="per_se")
    results = model.run(splits, se_ids)

    # Multitask mode
    model_mt = CatBoostPredictor(feature_set, mode="multitask")
    results_mt = model_mt.run_multitask(splits, se_ids)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import StandardScaler, LabelEncoder

from configs.config import CATBOOST_PARAMS, RANDOM_SEED
from features.pair_operators import make_pair_features
from evaluation.metrics import compute_metrics


class CatBoostPredictor:
    """
    CatBoost classifier — per-SE or multitask.

    Parameters
    ----------
    feature_set  : DrugFeatureSet
    operator     : pair combination operator
    feature_mode : "target_only" | "fp_only" | "target+fp"
    mode         : "per_se" | "multitask"
    params       : CatBoost params dict
    scale        : whether to StandardScale features (per_se mode only)
    """

    def __init__(self, feature_set, operator="hadamard_absdiff",
                 feature_mode="target+fp", mode="per_se",
                 params=None, scale=False):
        self.feature_set  = feature_set
        self.operator     = operator
        self.feature_mode = feature_mode
        self.mode         = mode
        self.params       = params or CATBOOST_PARAMS.copy()
        self.scale        = scale
        self.models_      = {}
        self.scalers_     = {}
        self.le_          = LabelEncoder()   # for multitask SE encoding

    def _get_X(self, df: pd.DataFrame) -> np.ndarray:
        zi = self.feature_set.get(df["drug_1"].tolist(), mode=self.feature_mode)
        zj = self.feature_set.get(df["drug_2"].tolist(), mode=self.feature_mode)
        return make_pair_features(zi, zj, operator=self.operator)

    # ── Per-SE mode ───────────────────────────────────────────────────────────

    def fit_one(self, se_id: str, train_df: pd.DataFrame,
                val_df: pd.DataFrame) -> CatBoostClassifier:
        X_tr = self._get_X(train_df)
        y_tr = train_df["label"].values
        X_vl = self._get_X(val_df)
        y_vl = val_df["label"].values

        if self.scale:
            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X_tr)
            X_vl   = scaler.transform(X_vl)
            self.scalers_[se_id] = scaler

        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        params = self.params.copy()
        params["class_weights"] = [1.0, n_neg / max(n_pos, 1)]

        clf = CatBoostClassifier(**params)
        clf.fit(Pool(X_tr, y_tr), eval_set=Pool(X_vl, y_vl),
                early_stopping_rounds=50, use_best_model=True)
        self.models_[se_id] = clf
        return clf

    def predict_proba(self, se_id: str, test_df: pd.DataFrame) -> np.ndarray:
        X = self._get_X(test_df)
        if self.scale and se_id in self.scalers_:
            X = self.scalers_[se_id].transform(X)
        return self.models_[se_id].predict_proba(X)[:, 1]

    def run(self, splits: dict, se_ids=None,
            verbose_every=50) -> pd.DataFrame:
        if se_ids is None:
            se_ids = list(splits.keys())
        rows = []
        for i, se_id in enumerate(se_ids):
            split = splits[se_id]
            if len(split.train) == 0 or split.train["label"].nunique() < 2:
                continue
            self.fit_one(se_id, split.train, split.val)
            proba   = self.predict_proba(se_id, split.test)
            y_true  = split.test["label"].values
            metrics = compute_metrics(y_true, proba)
            rows.append({"se_id": se_id, "model": "catboost",
                         "operator": self.operator, "feature_mode": self.feature_mode,
                         "strategy": split.strategy, **metrics})
            if (i + 1) % verbose_every == 0:
                print(f"  [{i+1}/{len(se_ids)}] {se_id} "
                      f"AUROC={metrics['auroc']:.3f} AP={metrics['ap']:.3f}")
        return pd.DataFrame(rows)

    # ── Multitask mode ────────────────────────────────────────────────────────

    def run_multitask(self, splits: dict, se_ids=None,
                      verbose_every=1) -> pd.DataFrame:
        """
        Train ONE CatBoost model across all SEs, with SE id as a categorical feature.
        Evaluate per SE on test sets.
        """
        if se_ids is None:
            se_ids = list(splits.keys())

        # Assemble combined train / val datasets
        tr_Xs, tr_ys, tr_ses = [], [], []
        vl_Xs, vl_ys, vl_ses = [], [], []

        for se_id in se_ids:
            split = splits[se_id]
            if len(split.train) == 0:
                continue
            X_tr = self._get_X(split.train)
            X_vl = self._get_X(split.val)
            se_col_tr = np.full(len(X_tr), se_id, dtype=object).reshape(-1, 1)
            se_col_vl = np.full(len(X_vl), se_id, dtype=object).reshape(-1, 1)
            tr_Xs.append(np.hstack([X_tr, se_col_tr]))
            tr_ys.append(split.train["label"].values)
            vl_Xs.append(np.hstack([X_vl, se_col_vl]))
            vl_ys.append(split.val["label"].values)

        X_train = np.vstack(tr_Xs)
        y_train = np.concatenate(tr_ys)
        X_val   = np.vstack(vl_Xs)
        y_val   = np.concatenate(vl_ys)

        # Last column is the categorical SE feature
        cat_idx = [X_train.shape[1] - 1]
        params  = self.params.copy()

        clf = CatBoostClassifier(**params, cat_features=cat_idx)
        clf.fit(Pool(X_train, y_train, cat_features=cat_idx),
                eval_set=Pool(X_val, y_val, cat_features=cat_idx),
                early_stopping_rounds=50, use_best_model=True)
        self.models_["multitask"] = clf

        # Evaluate per SE
        rows = []
        for se_id in se_ids:
            split  = splits.get(se_id)
            if split is None or len(split.test) == 0:
                continue
            X_te   = self._get_X(split.test)
            se_col = np.full(len(X_te), se_id, dtype=object).reshape(-1, 1)
            X_te_  = np.hstack([X_te, se_col])
            pool   = Pool(X_te_, cat_features=cat_idx)
            proba  = clf.predict_proba(pool)[:, 1]
            y_true = split.test["label"].values
            metrics = compute_metrics(y_true, proba)
            rows.append({"se_id": se_id, "model": "catboost_multitask",
                         "operator": self.operator, "feature_mode": self.feature_mode,
                         "strategy": split.strategy, **metrics})

        return pd.DataFrame(rows)
