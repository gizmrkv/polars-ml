from pathlib import Path

import optuna
import polars as pl
from polars import DataFrame
from sklearn.model_selection import KFold

from polars_ml.pipeline.pipeline import Pipeline


def test_lightgbm_basic(tmp_path: Path):
    df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 4.0]})

    pipeline = Pipeline().gbdt.lightgbm(
        "y",
        "pred",
        params={"objective": "regression", "verbose": -1},
        num_boost_round=10,
    )

    result = pipeline.fit_transform(df)

    assert "pred" in result.columns
    assert result["pred"].dtype == pl.Float64


def test_lightgbm_save(tmp_path: Path):
    fit_dir = tmp_path / "lgb_fit"
    df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0] * 10, "y": [1.0, 2.0, 3.0, 4.0] * 10})

    pipeline = Pipeline().gbdt.lightgbm(
        "y",
        "pred",
        params={"objective": "regression", "verbose": -1},
        fit_dir=fit_dir,
        num_boost_round=5,
    )

    pipeline.fit(df)

    assert fit_dir.exists()
    assert (fit_dir / "model.txt").exists()
    assert (fit_dir / "params.json").exists()


def test_lightgbm_tuner_basic():
    df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0] * 5, "y": [1.0, 2.0, 3.0, 4.0] * 5})

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pipeline = Pipeline().gbdt.lightgbm_tuner(
        "y",
        "pred",
        params={"objective": "regression", "metric": "l2", "verbose": -1},
        time_budget=1,
        num_boost_round=10,
    )

    result = pipeline.fit_transform(df)
    assert "pred" in result.columns


def test_lightgbm_tuner_cv_basic():
    df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0] * 10, "y": [1.0, 2.0, 3.0, 4.0] * 10})

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    kf = KFold(n_splits=3)

    pipeline = Pipeline().gbdt.lightgbm_tuner_cv(
        "y",
        "pred",
        params={"objective": "regression", "metric": "l2", "verbose": -1},
        time_budget=1,
        num_boost_round=5,
        folds=kf,
    )

    result = pipeline.fit_transform(df)

    assert "pred_cv_0" in result.columns
    assert "pred_cv_1" in result.columns
    assert "pred_cv_2" in result.columns
