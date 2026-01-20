import polars as pl
import polars.selectors as cs
import pytest

from polars_ml.metrics.binary_classification import BinaryClassificationMetrics


def test_binary_classification_metrics_single_pred():
    df = pl.DataFrame({"target": [0, 1, 0, 1], "pred": [0.1, 0.9, 0.2, 0.8]})

    metrics = BinaryClassificationMetrics(y_true="target", y_preds="pred")
    metrics.fit(df)
    result = metrics.transform(df)

    assert result.columns == ["prediction", "metric", "value"]
    assert result.filter(pl.col("prediction") == "pred").height > 0
    assert "log_loss" in result["metric"].to_list()


def test_binary_classification_metrics_multi_preds():
    df = pl.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "pred1": [0.1, 0.9, 0.2, 0.8],
            "pred2": [0.2, 0.8, 0.3, 0.7],
        }
    )

    metrics = BinaryClassificationMetrics(y_true="target", y_preds=["pred1", "pred2"])
    metrics.fit(df)
    result = metrics.transform(df)

    assert result.columns == ["prediction", "metric", "value"]
    predictions = result["prediction"].unique().to_list()
    assert "pred1" in predictions
    assert "pred2" in predictions


def test_binary_classification_metrics_with_by():
    df = pl.DataFrame(
        {
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "pred": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
            "group": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }
    )

    metrics = BinaryClassificationMetrics(y_true="target", y_preds="pred", by="group")
    metrics.fit(df)
    result = metrics.transform(df)

    assert result.columns == ["by", "prediction", "metric", "value"]
    groups = result["by"].unique().to_list()
    assert "A" in groups
    assert "B" in groups


def test_binary_classification_metrics_selector():
    df = pl.DataFrame(
        {
            "target": [0, 1, 0, 1],
            "pred1": [0.1, 0.9, 0.2, 0.8],
            "pred2": [0.2, 0.8, 0.3, 0.7],
            "other": [0, 1, 0, 1],
        }
    )

    metrics = BinaryClassificationMetrics(
        y_true="target", y_preds=cs.starts_with("pred")
    )
    metrics.fit(df)
    result = metrics.transform(df)

    predictions = result["prediction"].unique().to_list()
    assert "pred1" in predictions
    assert "pred2" in predictions
    assert "other" not in predictions


def test_binary_classification_metrics_constant_target():
    df = pl.DataFrame(
        {
            "target": [1, 1, 0, 1],
            "pred": [0.9, 0.8, 0.1, 0.7],
            "group": ["A", "A", "B", "B"],
        }
    )

    metrics = BinaryClassificationMetrics(y_true="target", y_preds="pred", by="group")
    metrics.fit(df)
    result = metrics.transform(df)

    groups = result["by"].unique().to_list()
    assert "A" not in groups
    assert "B" in groups
