from __future__ import annotations

import numpy as np
import xgboost as xgb
from numpy.typing import NDArray
from polars import DataFrame

from polars_ml.gbdt.xgboost_ import XGBoost


def test_xgboost_default_flow() -> None:
    df = DataFrame(
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


def test_base_xgboost_override() -> None:
    class CustomXGB(XGBoost):
        def fit(self, data: DataFrame) -> CustomXGB:
            import polars.selectors as cs

            if self.features_selector is None:
                label_columns = data.lazy().select(self.label).collect_schema().names()
                self.features_selector = cs.exclude(*label_columns)

            self.feature_names = data.select(self.features_selector).columns

            dtrain, _ = self.make_train_valid_sets(data)
            self.booster = xgb.train(self.params, dtrain, num_boost_round=5)
            return self

        def get_booster(self) -> xgb.Booster:
            return self.booster

        def create_dmatrix(self, data: DataFrame) -> xgb.DMatrix:
            dm = super().create_dmatrix(data)
            dm.set_info(base_margin=np.zeros(len(data)))
            return dm

        def predict(self, data: DataFrame) -> NDArray:
            return np.zeros(len(data))

    df = DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]})

    model = CustomXGB({"max_depth": 2}, "target")
    model.fit(df)
    result = model.transform(df)

    assert "prediction" in result.columns
    assert (result["prediction"] == 0).all()


def test_xgboost_feature_consistency() -> None:
    df_train = DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]})
    df_test = DataFrame({"f1": [1, 2, 3], "extra": [10, 20, 30], "target": [0, 1, 0]})

    model = XGBoost({"max_depth": 2}, label="target")
    model.fit(df_train)

    result = model.transform(df_test)

    assert "prediction" in result.columns
    assert len(result.columns) == 1
    assert "extra" not in result.columns
