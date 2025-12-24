from __future__ import annotations

from typing import Any

import catboost
import numpy as np
import polars as pl
from numpy.typing import NDArray

from polars_ml.gbdt.catboost_ import CatBoost


def test_catboost_default_flow():
    df = pl.DataFrame(
        {"f1": [1, 2, 3, 4, 5], "f2": [10, 20, 30, 40, 50], "target": [0, 1, 0, 1, 0]}
    )

    model = CatBoost({"iterations": 10, "verbose": False}, label="target")

    model.fit(df)
    result = model.transform(df)

    assert "prediction" in result.columns
    assert len(result) == 5


def test_base_catboost_override():
    class CustomCatBoost(CatBoost):
        def fit(self, data: pl.DataFrame) -> CustomCatBoost:
            import catboost

            pool = self.create_pool(data)
            params = dict(self.params)
            params.setdefault("iterations", 5)
            params.setdefault("logging_level", "Silent")
            self.model = catboost.CatBoost(params)
            self.model.fit(pool)
            return self

        def get_booster(self) -> catboost.CatBoost:
            return self.model

        def predict(self, data: pl.DataFrame) -> NDArray:
            # Custom prediction logic
            return np.zeros(len(data))

    df = pl.DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]})

    model = CustomCatBoost({"depth": 2}, "target")
    model.fit(df)
    result = model.transform(df)

    assert "prediction" in result.columns
    assert (result["prediction"] == 0).all()


def test_catboost_feature_consistency():
    df_train = pl.DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]})
    df_test = pl.DataFrame(
        {"f1": [1, 2, 3], "extra": [10, 20, 30], "target": [0, 1, 0]}
    )

    model = CatBoost({"depth": 2, "iterations": 5, "verbose": False}, label="target")
    model.fit(df_train)

    # Should not raise error even with 'extra' column
    result = model.transform(df_test)

    assert "prediction" in result.columns
    assert len(result.columns) == 4  # f1, extra, target, prediction
    assert "extra" in result.columns
