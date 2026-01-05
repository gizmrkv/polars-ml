import polars as pl
import pytest

from polars_ml import Pipeline
from polars_ml.base import HasFeatureImportance, Transformer
from polars_ml.optimize import OptunaOptimizer


class MockModel(Transformer, HasFeatureImportance):
    def __init__(self, param1=1):
        self.param1 = param1

    def fit(self, data, **more_data):
        self.data = data
        return self

    def transform(self, data):
        return data.with_columns(pl.lit(self.param1).alias("pred"))

    def get_feature_importance(self):
        return pl.DataFrame({"feature": ["a"], "importance": [self.param1]})


def model_fn(param1=1, trial=None):
    return MockModel(param1=param1)


def objective_fn(model, data, trial=None, **more_data):
    # Simulate evaluation
    return model.param1


def test_optuna_optimizer_basic():
    search_space = {"param1": {"min": 1, "max": 10}}

    df = pl.DataFrame({"a": [1, 2, 3]})

    optimizer = OptunaOptimizer(
        model_fn=model_fn,
        objective_fn=objective_fn,
        search_space=search_space,
        n_trials=5,
        study_name="test_study_basic",
        storage="./test_journal_basic.log",
        load_if_exists=True,
        is_higher_better=True,
    )

    optimizer.fit(df)
    assert hasattr(optimizer, "best_params")
    assert "param1" in optimizer.best_params
    assert 1 <= optimizer.best_params["param1"] <= 10

    res = optimizer.transform(df)
    assert "pred" in res.columns
    assert res["pred"][0] == optimizer.best_params["param1"]

    importance = optimizer.get_feature_importance()
    assert importance["importance"][0] == optimizer.best_params["param1"]


def test_pipeline_integration():
    search_space = {"param1": {"min": 1, "max": 10}}

    df = pl.DataFrame({"a": [1, 2, 3]})

    pipeline = Pipeline().optimize.optuna(
        model_fn=model_fn,
        objective_fn=objective_fn,
        search_space=search_space,
        n_trials=3,
        study_name="test_study_pipeline",
        storage="./test_journal_pipeline.log",
        load_if_exists=True,
        is_higher_better=False,
    )

    pipeline.fit(df)
    res = pipeline.transform(df)
    assert "pred" in res.columns

    importance = pipeline.get_feature_importance()
    assert "importance" in importance.columns


def test_more_data_passing():
    def objective_with_more_data(model, data, trial=None, **more_data):
        assert "valid" in more_data
        return model.param1

    search_space = {"param1": {"value": 5}}
    df = pl.DataFrame({"a": [1, 2, 3]})
    valid_df = pl.DataFrame({"a": [4, 5, 6]})

    optimizer = OptunaOptimizer(
        model_fn=model_fn,
        objective_fn=objective_with_more_data,
        search_space=search_space,
        n_trials=1,
        storage="./test_journal_more.log",
    )

    optimizer.fit(df, valid=valid_df)
    assert optimizer.best_params["param1"] == 5
