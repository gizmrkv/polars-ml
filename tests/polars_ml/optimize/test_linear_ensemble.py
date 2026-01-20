import numpy as np
import polars as pl
import pytest

from polars_ml.optimize.linear_ensemble import LinearEnsemble
from polars_ml.pipeline import Pipeline


def test_linear_ensemble_ridge():
    # Test Ridge regression (l1_ratio=0)
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

    le = LinearEnsemble(
        pred_columns=["p1", "p2"],
        target_column="target",
        alpha=0.01,
        l1_ratio=0.0,  # Ridge
        positive=True,
    )

    le.fit(df_train, validation=df_val)
    weights = le.get_weights()

    # Check weights are non-negative
    assert all(w >= 0 for w in weights.values())
    # Check weights are close to 0.3 and 0.7
    assert np.isclose(weights["p1"], 0.3, atol=0.1)
    assert np.isclose(weights["p2"], 0.7, atol=0.1)

    # Transform
    df_transformed = le.transform(df_val)
    assert "ensemble" in df_transformed.columns


def test_linear_ensemble_lasso():
    # Test Lasso regression (l1_ratio=1) for sparsity
    np.random.seed(42)
    n = 100
    p1 = np.random.rand(n)
    p2 = np.random.rand(n)
    p3 = np.random.rand(n)  # Irrelevant predictor
    y = 0.5 * p1 + 0.5 * p2  # p3 should get zero weight

    df_train = pl.DataFrame({"p1": p1, "p2": p2, "p3": p3, "target": y})

    # Create validation data
    p1_val = np.random.rand(n)
    p2_val = np.random.rand(n)
    p3_val = np.random.rand(n)
    y_val = 0.5 * p1_val + 0.5 * p2_val
    df_val = pl.DataFrame({"p1": p1_val, "p2": p2_val, "p3": p3_val, "target": y_val})

    le = LinearEnsemble(
        pred_columns=["p1", "p2", "p3"],
        target_column="target",
        alpha=0.1,
        l1_ratio=1.0,  # Lasso
        positive=True,
    )

    le.fit(df_train, validation=df_val)
    weights = le.get_weights()

    # Check that p3 has very small or zero weight (sparsity)
    assert weights["p3"] < 0.1


def test_linear_ensemble_elastic_net():
    # Test ElasticNet (l1_ratio=0.5)
    np.random.seed(42)
    n = 100
    p1 = np.random.rand(n)
    p2 = np.random.rand(n)
    y = 0.4 * p1 + 0.6 * p2

    df_train = pl.DataFrame({"p1": p1, "p2": p2, "target": y})

    # Create validation data
    p1_val = np.random.rand(n)
    p2_val = np.random.rand(n)
    y_val = 0.4 * p1_val + 0.6 * p2_val
    df_val = pl.DataFrame({"p1": p1_val, "p2": p2_val, "target": y_val})

    le = LinearEnsemble(
        pred_columns=["p1", "p2"],
        target_column="target",
        alpha=0.01,
        l1_ratio=0.5,  # ElasticNet
        positive=True,
    )

    le.fit(df_train, validation=df_val)
    weights = le.get_weights()

    # Check weights are non-negative
    assert all(w >= 0 for w in weights.values())


def test_linear_ensemble_pipeline():
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

    pipeline = Pipeline().optimize.linear_ensemble(
        pred_columns=["p1", "p2"],
        target_column="target",
        alpha=0.01,
        output_column="ensemble_pred",
    )

    df_transformed = pipeline.fit_transform(df_train, validation=df_val)
    assert "ensemble_pred" in df_transformed.columns


def test_linear_ensemble_with_intercept():
    # Test with fit_intercept=True
    np.random.seed(42)
    n = 100
    p1 = np.random.rand(n)
    p2 = np.random.rand(n)
    y = 0.3 * p1 + 0.7 * p2 + 0.1  # With intercept

    df_train = pl.DataFrame({"p1": p1, "p2": p2, "target": y})

    # Create validation data
    p1_val = np.random.rand(n)
    p2_val = np.random.rand(n)
    y_val = 0.3 * p1_val + 0.7 * p2_val + 0.1
    df_val = pl.DataFrame({"p1": p1_val, "p2": p2_val, "target": y_val})

    le = LinearEnsemble(
        pred_columns=["p1", "p2"],
        target_column="target",
        alpha=0.01,
        fit_intercept=True,
        positive=True,
    )

    le.fit(df_train, validation=df_val)
    intercept = le.get_intercept()

    # Check intercept is close to 0.1
    assert np.isclose(intercept, 0.1, atol=0.15)
