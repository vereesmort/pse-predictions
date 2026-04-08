"""
models/lightgbm_model.py
========================
LightGBM-based polypharmacy side effect predictor.

One binary classifier per side effect, trained independently.
Supports all drug pair combination operators and feature modes.

Usage
-----
    from models.lightgbm_model import LGBMPredictor
    model = LGBMPredictor(feature_set, operator="hadamard_absdiff",
                          feature_mode="target+fp")
    results = model.run(splits, se_ids)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

from configs.config import LGBM_PARAMS, RANDOM_SEED
from features.pair_operators import make_pair_features
from evaluation.metrics import compute_metrics


class LGBMPredictor:
    """
    Per-SE LightGBM classifier.

    Parameters
    ----------
    feature_set  : DrugFeatureSet  (from features/drug_features.py)
    operator     : pair combination operator  (from features/pair_operators.py)
    feature_mode : "target_only" | "fp_only" | "target+fp"
    params       : LightGBM params dict (default from config)
    scale        : whether to StandardScale pair features before fitting
    """

    def __init__(self, feature_set, operator="hadamard_absdiff",
                 feature_mode="target+fp", params=None, scale=True):
        self.feature_set  = feature_set
        self.operator     = operator
        self.feature_mode = feature_mode
        self.params       = params or LGBM_PARAMS.copy()
        self.scale        = scale
        self.models_      = {}   # se_id → trained lgb.Booster
        self.scalers_     = {}   # se_id → StandardScaler

    # ── Feature construction ───────────────────────────────────────────────────

    def _get_X(self, df: pd.DataFrame) -> np.ndarray:
        zi = self.feature_set.get(df["drug_1"].tolist(), mode=self.feature_mode)
        zj = self.feature_set.get(df["drug_2"].tolist(), mode=self.feature_mode)
        return make_pair_features(zi, zj, operator=self.operator)

    # ── Train one SE ──────────────────────────────────────────────────────────

    def fit_one(self, se_id: str, train_df: pd.DataFrame,
                val_df: pd.DataFrame) -> lgb.Booster:
        X_tr = self._get_X(train_df)
        y_tr = train_df["label"].values
        X_vl = self._get_X(val_df)
        y_vl = val_df["label"].values

        if self.scale:
            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X_tr)
            X_vl   = scaler.transform(X_vl)
            self.scalers_[se_id] = scaler

        # Adjust class weight for imbalance
        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        params = self.params.copy()
        params["scale_pos_weight"] = n_neg / max(n_pos, 1)

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_vl, label=y_vl, reference=dtrain)

        callbacks = [lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(period=-1)]
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=params.pop("n_estimators", 500),
            valid_sets=[dval],
            callbacks=callbacks,
        )
        self.models_[se_id] = booster
        return booster

    def predict_proba(self, se_id: str, test_df: pd.DataFrame) -> np.ndarray:
        X = self._get_X(test_df)
        if self.scale and se_id in self.scalers_:
            X = self.scalers_[se_id].transform(X)
        return self.models_[se_id].predict(X)

    # ── Run over all SEs ──────────────────────────────────────────────────────

    def run(self, splits: dict, se_ids=None,
            verbose_every=50) -> pd.DataFrame:
        """
        Train and evaluate over all (or specified) side effects.

        splits : dict {se_id: SplitResult}  (from preprocessing/splitting.py)
        Returns a DataFrame with per-SE metrics.
        """
        if se_ids is None:
            se_ids = list(splits.keys())

        rows = []
        for i, se_id in enumerate(se_ids):
            split = splits[se_id]
            if len(split.train) == 0 or split.train["label"].nunique() < 2:
                continue

            booster  = self.fit_one(se_id, split.train, split.val)
            proba    = self.predict_proba(se_id, split.test)
            y_true   = split.test["label"].values
            metrics  = compute_metrics(y_true, proba)

            row = {"se_id": se_id, "model": "lightgbm",
                   "operator": self.operator, "feature_mode": self.feature_mode,
                   "strategy": split.strategy, **metrics}
            rows.append(row)

            if (i + 1) % verbose_every == 0:
                print(f"  [{i+1}/{len(se_ids)}] {se_id} "
                      f"AUROC={metrics['auroc']:.3f} AP={metrics['ap']:.3f}")

        return pd.DataFrame(rows)
