from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb
from numpy.typing import NDArray

from polars_ml.gbdt.xgboost_ import BaseXGBoost, XGBoost


def test_xgboost_default_flow():
    df = pl.DataFrame(
        {"f1": [1, 2, 3, 4, 5], "f2": [10, 20, 30, 40, 50], "target": [0, 1, 0, 1, 0]}
    )

    class MyXGB(XGBoost):
        def get_train_params(self) -> dict[str, Any]:
            return {"num_boost_round": 10}

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 3,
    }

    model = MyXGB(label="target", **params)

    model.fit(df)
    result = model.transform(df)

    assert "prediction" in result.columns
    assert len(result) == 5


def test_base_xgboost_override():
    class CustomXGB(BaseXGBoost):
        def fit(self, data: pl.DataFrame) -> CustomXGB:
            dtrain, _ = self.make_train_valid_sets(data)
            self.booster = xgb.train(self.params, dtrain, num_boost_round=5)
            return self

        def get_booster(self) -> xgb.Booster:
            return self.booster

        def create_train(self, data: pl.DataFrame) -> xgb.DMatrix:
            dm = super().create_train(data)
            dm.set_info(base_margin=np.zeros(len(data)))
            return dm

        def predict(self, data: pl.DataFrame) -> NDArray:
            # Custom prediction logic
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

    model = XGBoost(label="target", max_depth=2)
    model.fit(df_train)

    # Should not raise error even with 'extra' column
    result = model.transform(df_test)

    assert "prediction" in result.columns
    assert len(result.columns) == 4  # f1, extra, target, prediction
    assert "extra" in result.columns
