from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb
from numpy.typing import NDArray

from polars_ml.gbdt.xgboost_ import XGBoost


def test_xgboost_default_flow():
    df = pl.DataFrame(
        {"f1": [1, 2, 3, 4, 5], "f2": [10, 20, 30, 40, 50], "target": [0, 1, 0, 1, 0]}
    )

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 3,
    }

    model = XGBoost(params, label="target")

    model.fit(df)
    result = model.transform(df)

    assert "prediction" in result.columns
    assert len(result) == 5


def test_base_xgboost_override():
    class CustomXGB(XGBoost):
        def fit(self, data: pl.DataFrame) -> CustomXGB:
            import polars.selectors as cs

            if self.features_selector is None:
                label_cols = data.lazy().select(self.label).collect_schema().names()
                self.features_selector = cs.exclude(*label_cols)

            self.feature_names = data.select(self.features_selector).columns

            dtrain, _ = self.make_train_valid_sets(data)
            self.booster = xgb.train(self.params, dtrain, num_boost_round=5)
            return self

        def get_booster(self) -> xgb.Booster:
            return self.booster

        def create_dmatrix(self, data: pl.DataFrame) -> xgb.DMatrix:
            dm = super().create_dmatrix(data)
            dm.set_info(base_margin=np.zeros(len(data)))
            return dm

        def predict(self, data: pl.DataFrame) -> NDArray:
            return np.zeros(len(data))

    df = pl.DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]})

    model = CustomXGB({"max_depth": 2}, "target")
    model.fit(df)
    result = model.transform(df)

    assert "prediction" in result.columns
    assert (result["prediction"] == 0).all()


def test_xgboost_feature_consistency():
    df_train = pl.DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]})
    df_test = pl.DataFrame(
        {"f1": [1, 2, 3], "extra": [10, 20, 30], "target": [0, 1, 0]}
    )

    model = XGBoost({"max_depth": 2}, label="target")
    model.fit(df_train)

    result = model.transform(df_test)

    assert "prediction" in result.columns
    assert len(result.columns) == 4
    assert "extra" in result.columns
