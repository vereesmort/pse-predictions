"""
models/xgboost_model.py
=======================
XGBoost-based polypharmacy side effect predictor.

Structurally identical interface to LGBMPredictor for easy swapping.

Usage
-----
    from models.xgboost_model import XGBPredictor
    model = XGBPredictor(feature_set, operator="hadamard_absdiff",
                         feature_mode="target+fp")
    results = model.run(splits, se_ids)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from configs.config import XGB_PARAMS, RANDOM_SEED
from features.pair_operators import make_pair_features
from evaluation.metrics import compute_metrics


class XGBPredictor:
    """
    Per-SE XGBoost binary classifier.

    Parameters
    ----------
    feature_set  : DrugFeatureSet
    operator     : pair combination operator
    feature_mode : "target_only" | "fp_only" | "target+fp"
    params       : XGBoost params dict
    scale        : whether to StandardScale features
    """

    def __init__(self, feature_set, operator="hadamard_absdiff",
                 feature_mode="target+fp", params=None, scale=True):
        self.feature_set  = feature_set
        self.operator     = operator
        self.feature_mode = feature_mode
        self.params       = params or XGB_PARAMS.copy()
        self.scale        = scale
        self.models_      = {}
        self.scalers_     = {}

    def _get_X(self, df: pd.DataFrame) -> np.ndarray:
        zi = self.feature_set.get(df["drug_1"].tolist(), mode=self.feature_mode)
        zj = self.feature_set.get(df["drug_2"].tolist(), mode=self.feature_mode)
        return make_pair_features(zi, zj, operator=self.operator)

    def fit_one(self, se_id: str, train_df: pd.DataFrame,
                val_df: pd.DataFrame) -> xgb.XGBClassifier:
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
        params["scale_pos_weight"] = n_neg / max(n_pos, 1)

        clf = xgb.XGBClassifier(**params)
        clf.fit(X_tr, y_tr,
                eval_set=[(X_vl, y_vl)],
                verbose=False)
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
            rows.append({"se_id": se_id, "model": "xgboost",
                         "operator": self.operator, "feature_mode": self.feature_mode,
                         "strategy": split.strategy, **metrics})
            if (i + 1) % verbose_every == 0:
                print(f"  [{i+1}/{len(se_ids)}] {se_id} "
                      f"AUROC={metrics['auroc']:.3f} AP={metrics['ap']:.3f}")
        return pd.DataFrame(rows)
