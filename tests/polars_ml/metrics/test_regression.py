import polars as pl
import polars.selectors as cs
import pytest

from polars_ml.metrics.regression import RegressionMetrics


def test_regression_metrics_single_pred():
    df = pl.DataFrame({"target": [1.0, 2.0, 3.0], "pred": [1.1, 1.9, 3.2]})

    metrics = RegressionMetrics(y_true="target", y_preds="pred")
    result = metrics.transform(df)

    assert result.columns == ["prediction", "metric", "value"]
    assert result.filter(pl.col("prediction") == "pred").height > 0
    assert "mse" in result["metric"].to_list()


def test_regression_metrics_multi_preds():
    df = pl.DataFrame(
        {"target": [1.0, 2.0, 3.0], "pred1": [1.1, 1.9, 3.2], "pred2": [1.2, 2.1, 2.9]}
    )

    metrics = RegressionMetrics(y_true="target", y_preds=["pred1", "pred2"])
    result = metrics.transform(df)

    assert result.columns == ["prediction", "metric", "value"]
    predictions = result["prediction"].unique().to_list()
    assert "pred1" in predictions
    assert "pred2" in predictions
    assert result.filter(pl.col("prediction") == "pred1").height > 0
    assert result.filter(pl.col("prediction") == "pred2").height > 0


def test_regression_metrics_with_by():
    df = pl.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, 4.0],
            "pred": [1.1, 1.9, 3.1, 3.9],
            "group": ["A", "A", "B", "B"],
        }
    )

    metrics = RegressionMetrics(y_true="target", y_preds="pred", by="group")
    result = metrics.transform(df)

    assert result.columns == ["by", "prediction", "metric", "value"]
    groups = result["by"].unique().to_list()
    assert "A" in groups
    assert "B" in groups
    assert (
        result.filter((pl.col("by") == "A") & (pl.col("prediction") == "pred")).height
        > 0
    )


def test_regression_metrics_multi_preds_with_by():
    df = pl.DataFrame(
        {
            "target": [1.0, 2.0, 3.0, 4.0],
            "pred1": [1.1, 1.9, 3.1, 3.9],
            "pred2": [1.2, 2.1, 2.9, 4.1],
            "group": ["A", "A", "B", "B"],
        }
    )

    metrics = RegressionMetrics(y_true="target", y_preds=["pred1", "pred2"], by="group")
    result = metrics.transform(df)

    assert result.columns == ["by", "prediction", "metric", "value"]
    groups = result["by"].unique().to_list()
    assert "A" in groups
    assert "B" in groups
    predictions = result["prediction"].unique().to_list()
    assert "pred1" in predictions
    assert "pred2" in predictions

    assert (
        result.filter((pl.col("by") == "A") & (pl.col("prediction") == "pred1")).height
        > 0
    )
    assert (
        result.filter((pl.col("by") == "B") & (pl.col("prediction") == "pred2")).height
        > 0
    )


def test_regression_metrics_selector():
    df = pl.DataFrame(
        {
            "target": [1.0, 2.0, 3.0],
            "pred1": [1.1, 1.9, 3.2],
            "pred2": [1.2, 2.1, 2.9],
            "other": [0, 0, 0],
        }
    )

    # Use selector to pick columns starting with 'pred'
    metrics = RegressionMetrics(y_true="target", y_preds=cs.starts_with("pred"))
    result = metrics.transform(df)

    assert result.columns == ["prediction", "metric", "value"]
    predictions = result["prediction"].unique().to_list()
    assert "pred1" in predictions
    assert "pred2" in predictions
    assert "other" not in predictions


def test_regression_metrics_missing_y_true():
    df = pl.DataFrame(
        {
            "pred": [1.1, 1.9, 3.2],
        }
    )

    metrics = RegressionMetrics(y_true="target", y_preds="pred")
    with pytest.raises(ValueError, match="y_true column 'target' not found in data"):
        metrics.transform(df)
