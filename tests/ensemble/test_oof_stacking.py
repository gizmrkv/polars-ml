from typing import Self

import polars as pl
from polars import DataFrame

from polars_ml.base import Transformer
from polars_ml.ensemble.oof_stacking import Stacking
from polars_ml.model_selection import KFold


class MockModel(Transformer):
    def __init__(self, fold_id: int):
        self.fold_id = fold_id

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        # Return a constant value (fold_id) for each row
        return DataFrame({"pred": [float(self.fold_id)] * len(data)})


def test_stacking_basic():
    df = DataFrame({"x": range(10), "y": [0, 1] * 5})

    def model_fn(train_data, fold_idx):
        return MockModel(fold_idx)

    k_fold = KFold(n_splits=2, shuffle=False)
    stacking = Stacking(model_fn, k_fold)

    # Test fit_transform (OOF predictions)
    # Fold 0: valid_idx [0, 2, 4, 6, 8] -> pred 0.0
    # Fold 1: valid_idx [1, 3, 5, 7, 9] -> pred 1.0
    # Expected OOF: [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    oof = stacking.fit_transform(df)
    assert oof["pred"].to_list() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    # Test transform (Average predictions)
    # Average of fold 0 (0.0) and fold 1 (1.0) is 0.5
    avg_pred = stacking.transform(df)
    assert avg_pred["pred"].to_list() == [0.5] * 10


def test_stacking_aggs():
    df = DataFrame({"x": range(4), "y": [0] * 4})

    def model_fn(train_data, fold_idx):
        return MockModel(fold_idx)

    k_fold = KFold(n_splits=2, shuffle=False)
    # Test custom aggregation (max)
    stacking = Stacking(model_fn, k_fold, aggs_on_transform=pl.col("pred").max())
    stacking.fit(df)

    # Max of fold 0 (0.0) and fold 1 (1.0) is 1.0
    res = stacking.transform(df)
    assert res["pred"].to_list() == [1.0] * 4
