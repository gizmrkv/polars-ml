import polars as pl
import pytest
from polars import DataFrame

from polars_ml.pipeline.pipeline import Pipeline


def test_linear_regression():
    df = DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 6.0, 8.0, 10.0]})

    pipeline = Pipeline().linear.linear_regression("y", "pred")

    result = pipeline.fit_transform(df)

    assert "pred" in result.columns
    preds = result["pred"].to_numpy()
    targets = df["y"].to_numpy()
    assert ((preds - targets) ** 2).mean() < 1e-10


def test_logistic_regression():
    df = DataFrame(
        {"x1": [1.0, 2.0, 5.0, 6.0], "x2": [1.0, 1.0, 5.0, 5.0], "y": [0, 0, 1, 1]}
    )

    pipeline = Pipeline().linear.logistic_regression("y", "pred")

    result = pipeline.fit_transform(df)

    assert "pred" in result.columns
    assert result["pred"].dtype == pl.Int64
    assert result["pred"].to_list() == [0, 0, 1, 1]


def test_linear_feature_importance():
    df = DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [0.0, 0.0, 0.0], "y": [1.0, 2.0, 3.0]})

    pipeline = Pipeline().linear.linear_regression("y", "pred")

    pipeline.fit(df)
    importance = pipeline.get_feature_importance()

    assert "feature" in importance.columns
    assert "coefficient" in importance.columns

    x1_coef = importance.filter(pl.col("feature") == "x1")["coefficient"][0]
    x2_coef = importance.filter(pl.col("feature") == "x2")["coefficient"][0]

    assert abs(x1_coef - 1.0) < 1e-10
    assert abs(x2_coef) < 1e-10


def test_ridge_regression():
    df = DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.1, 1.9, 3.0]})

    pipeline = Pipeline().linear.ridge("y", "pred", alpha=0.1)

    result = pipeline.fit_transform(df)
    assert "pred" in result.columns


def test_lasso_regression():
    df = DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.1, 1.9, 3.0]})

    pipeline = Pipeline().linear.lasso("y", "pred", alpha=0.1)

    result = pipeline.fit_transform(df)
    assert "pred" in result.columns
