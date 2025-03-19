import tempfile
from typing import Mapping

import lightgbm as lgb
import optuna
import polars as pl
from polars import DataFrame

from polars_ml import Pipeline, PipelineComponent
from polars_ml.model_selection import train_test_split
from polars_ml.pipeline.testing import assert_component_valid


def test_pipeline_optimize_optuna_friedman1(
    test_data_friedman1: DataFrame,
):
    def make_model(
        learning_rate: float,
        num_leaves: int,
        max_depth: int,
        feature_fraction: float,
        min_data_in_leaf: int,
        bagigng_fraction: float,
        lambda_l1: float,
        lambda_l2: float,
        trial: optuna.Trial | None = None,
    ) -> Pipeline:
        return Pipeline().gbdt.lightgbm(
            pl.exclude("y"),
            "y",
            {
                "objective": "regression",
                "verbosity": -1,
                "learning_rate": learning_rate,
                "num_leaves": num_leaves,
                "max_depth": max_depth,
                "feature_fraction": feature_fraction,
                "min_data_in_leaf": min_data_in_leaf,
                "bagging_fraction": bagigng_fraction,
                "lambda_l1": lambda_l1,
                "lambda_l2": lambda_l2,
            },
            train_kwargs={"callbacks": [lgb.early_stopping(5)]},
            prediction_name="prediction",
        )

    def objective(
        model: PipelineComponent,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
        *,
        trial: optuna.Trial | None = None,
    ) -> float:
        assert isinstance(validation_data, DataFrame)

        model.fit(data, validation_data)
        valid_pred_df = model.transform(validation_data)

        return valid_pred_df.select(
            (pl.col("y") - pl.col("prediction")).abs().mean().alias("loss")
        ).item()

    train_idx, valid_idx = train_test_split(
        test_data_friedman1, 0.2, shuffle=True, seed=42
    )
    train_df = test_data_friedman1.select(pl.all().gather(train_idx))
    valid_df = test_data_friedman1.select(pl.all().gather(valid_idx))
    with tempfile.TemporaryDirectory() as tmpdir:
        assert_component_valid(
            Pipeline()
            .optimize.optuna(
                make_model,
                objective,
                {
                    "learning_rate": {"min": 0.001, "max": 0.2, "log": True},
                    "num_leaves": {"min": 10, "max": 100},
                    "max_depth": {"min": 3, "max": 12},
                    "feature_fraction": {"min": 0.5, "max": 1.0},
                    "min_data_in_leaf": {"min": 20, "max": 200},
                    "bagigng_fraction": {"min": 0.6, "max": 1.0},
                    "lambda_l1": {"min": 0.0, "max": 10.0},
                    "lambda_l2": {"min": 0.0, "max": 10.0},
                },
                storage=tmpdir + "/journal.log",
                is_higher_better=False,
                n_trials=5,
            )
            .select(
                (pl.col("y") - pl.col("prediction")).abs().mean().round(1).alias("mae")
            ),
            train_df,
            validation_data=valid_df,
        )
