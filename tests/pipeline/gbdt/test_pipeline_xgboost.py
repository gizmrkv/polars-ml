from pathlib import Path

import polars as pl
import pytest
from polars import DataFrame

from polars_ml.pipeline.pipeline import Pipeline


def test_xgboost_basic():
    df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})

    pipeline = Pipeline().gbdt.xgboost(
        target="y",
        prediction="pred",
        features=["x"],
        params={"objective": "reg:squarederror", "verbosity": 0},
        num_boost_round=10,
    )

    result = pipeline.fit_transform(df)

    assert "pred" in result.columns
    assert result["pred"].dtype == pl.Float32


def test_xgboost_save(tmp_path: Path):
    df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0] * 10, "y": [1.0, 2.0, 3.0, 4.0] * 10})

    pipeline = Pipeline().gbdt.xgboost(
        target="y",
        prediction="pred",
        features=["x"],
        params={"objective": "reg:squarederror", "verbosity": 0},
        fit_dir=tmp_path,
        num_boost_round=5,
    )

    pipeline.fit(df)

    assert tmp_path.exists()
    assert (tmp_path / "model.json").exists()
    assert (tmp_path / "params.json").exists()
