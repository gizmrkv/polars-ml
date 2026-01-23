import tempfile
from pathlib import Path

import polars as pl
from polars import DataFrame

from polars_ml.pipeline.pipeline import Pipeline


def test_catboost_basic():
    df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})

    pipeline = Pipeline().gbdt.catboost(
        target="y",
        prediction="pred",
        features=["x"],
        params={"loss_function": "RMSE", "verbose": False, "iterations": 10},
    )

    result = pipeline.fit_transform(df)

    assert "pred" in result.columns
    assert result["pred"].dtype == pl.Float64


def test_catboost_save(tmp_path: Path):
    df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0] * 10, "y": [1.0, 2.0, 3.0, 4.0] * 10})

    pipeline = Pipeline().gbdt.catboost(
        target="y",
        prediction="pred",
        features=["x"],
        params={"loss_function": "RMSE", "verbose": False, "iterations": 5},
        fit_dir=tmp_path,
    )

    pipeline.fit(df)

    assert tmp_path.exists()
    assert (tmp_path / "model.cbm").exists()
