import numpy as np
import polars as pl
import pytest
from sklearn.metrics import mean_squared_error

from polars_ml.optimize.weighted_average import WeightedAverage
from polars_ml.pipeline import Pipeline


def test_weighted_average_basic():
    # Create synthetic data
    # y = 0.3 * p1 + 0.7 * p2
    np.random.seed(42)
    n = 100
    p1 = np.random.rand(n)
    p2 = np.random.rand(n)
    y = 0.3 * p1 + 0.7 * p2 + np.random.normal(0, 0.01, n)

    df_train = pl.DataFrame({"p1": p1, "p2": p2, "target": y})

    # Create validation data
    p1_val = np.random.rand(n)
    p2_val = np.random.rand(n)
    y_val = 0.3 * p1_val + 0.7 * p2_val + np.random.normal(0, 0.01, n)
    df_val = pl.DataFrame({"p1": p1_val, "p2": p2_val, "target": y_val})

    wa = WeightedAverage(
        pred_columns=["p1", "p2"],
        target_column="target",
        sum_to_one=True,
        non_negative=True,
    )

    wa.fit(df_train, validation=df_val)
    weights = wa.get_weights()

    # Check weights sum to 1
    assert np.isclose(sum(weights.values()), 1.0)
    # Check weights are close to 0.3 and 0.7
    assert np.isclose(weights["p1"], 0.3, atol=0.05)
    assert np.isclose(weights["p2"], 0.7, atol=0.05)

    # Transform
    df_transformed = wa.transform(df_val)
    assert "weighted_average" in df_transformed.columns

    # Calculate MSE
    y_true = df_val["target"].to_numpy()
    y_pred = df_transformed["weighted_average"].to_numpy()
    mse = mean_squared_error(y_true, y_pred)

    # Compare with individual MSEs
    mse1 = mean_squared_error(y_true, df_val["p1"].to_numpy())
    mse2 = mean_squared_error(y_true, df_val["p2"].to_numpy())

    assert mse < mse1
    assert mse < mse2


def test_weighted_average_pipeline():
    np.random.seed(42)
    n = 100
    p1 = np.random.rand(n)
    p2 = np.random.rand(n)
    y = 0.5 * p1 + 0.5 * p2

    df_train = pl.DataFrame({"p1": p1, "p2": p2, "target": y})

    # Create validation data
    p1_val = np.random.rand(n)
    p2_val = np.random.rand(n)
    y_val = 0.5 * p1_val + 0.5 * p2_val
    df_val = pl.DataFrame({"p1": p1_val, "p2": p2_val, "target": y_val})

    pipeline = Pipeline().optimize.weighted_average(
        pred_columns=["p1", "p2"], target_column="target", output_column="ensemble"
    )

    df_transformed = pipeline.fit_transform(df_train, validation=df_val)
    assert "ensemble" in df_transformed.columns


def test_weighted_average_method_nelder_mead():
    # Nelder-Mead doesn't support constraints, so sum_to_one/non_negative might be ignored or fail
    # In our implementation we pass them to minimize.
    # Scipy should ignore them or raise warning/error depending on version.
    # Our implementation just passes them.

    np.random.seed(42)
    n = 100
    p1 = np.random.rand(n)
    y = 0.5 * p1

    df_train = pl.DataFrame({"p1": p1, "target": y})

    # Create validation data
    p1_val = np.random.rand(n)
    y_val = 0.5 * p1_val
    df_val = pl.DataFrame({"p1": p1_val, "target": y_val})

    # Using Nelder-Mead
    wa = WeightedAverage(
        pred_columns=["p1"],
        target_column="target",
        method="Nelder-Mead",
        sum_to_one=False,
        non_negative=False,
    )

    wa.fit(df_train, validation=df_val)
    weights = wa.get_weights()
    assert np.isclose(weights["p1"], 0.5, atol=1e-3)
