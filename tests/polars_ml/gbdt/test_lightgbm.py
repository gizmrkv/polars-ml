from __future__ import annotations

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray
from polars import DataFrame

from polars_ml.gbdt.lightgbm_ import BaseLightGBM, LightGBM


def test_lightgbm_default_flow() -> None:
    df = DataFrame(
        {"f1": [1, 2, 3, 4, 5], "f2": [10, 20, 30, 40, 50], "target": [0, 1, 0, 1, 0]}
    )

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }

    model = LightGBM(params, label="target")

    model.fit(df)
    result = model.transform(df)

    assert "prediction" in result.columns
    assert len(result) == 5


def test_base_lightgbm_override() -> None:
    class CustomLGB(BaseLightGBM):
        def fit(self, data: DataFrame) -> CustomLGB:
            train_dataset, _, _ = self.make_train_valid_sets(data)
            self.booster = lgb.train(self.params, train_dataset, num_boost_round=5)
            return self

        def get_booster(self) -> lgb.Booster:
            return self.booster

        def create_train(self, data: DataFrame) -> lgb.Dataset:
            ds = super().create_train(data)
            ds.custom_attr = "custom"
            return ds

        def predict(self, data: DataFrame) -> NDArray:
            return np.zeros(len(data))

    df = DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]})

    model = CustomLGB({"verbosity": -1}, "target")
    model.fit(df)
    result = model.transform(df)

    assert "prediction" in result.columns
    assert (result["prediction"] == 0).all()


def test_feature_consistency() -> None:
    df_train = DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]})
    df_test = DataFrame({"f1": [1, 2, 3], "extra": [10, 20, 30], "target": [0, 1, 0]})

    model = LightGBM({"verbosity": -1}, label="target")
    model.fit(df_train)

    result = model.transform(df_test)

    assert "prediction" in result.columns
    assert len(result.columns) == 1
    assert "extra" not in result.columns
