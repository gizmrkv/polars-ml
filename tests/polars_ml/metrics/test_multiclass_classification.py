import polars as pl
import polars.selectors as cs
import pytest

from polars_ml.metrics.multiclass_classification import MulticlassClassificationMetrics


def test_multiclass_classification_metrics_single_pred():
    df = pl.DataFrame({"target": [0, 1, 2, 0, 1, 2], "pred": [0, 1, 2, 1, 1, 2]})

    metrics = MulticlassClassificationMetrics(y_true="target", y_preds="pred")
    result = metrics.transform(df)

    assert result.columns == ["prediction", "metric", "value"]
    assert result.filter(pl.col("prediction") == "pred").height > 0
    assert "accuracy" in result["metric"].to_list()
    assert "f1_macro" in result["metric"].to_list()


def test_multiclass_classification_metrics_multi_preds():
    df = pl.DataFrame(
        {
            "target": [0, 1, 2, 0, 1, 2],
            "pred1": [0, 1, 2, 1, 1, 2],
            "pred2": [0, 1, 1, 0, 1, 2],
        }
    )

    metrics = MulticlassClassificationMetrics(
        y_true="target", y_preds=["pred1", "pred2"]
    )
    result = metrics.transform(df)

    assert result.columns == ["prediction", "metric", "value"]
    predictions = result["prediction"].unique().to_list()
    assert "pred1" in predictions
    assert "pred2" in predictions


def test_multiclass_classification_metrics_with_by():
    df = pl.DataFrame(
        {
            "target": [0, 1, 2, 0, 1, 2],
            "pred": [0, 1, 2, 0, 1, 2],
            "group": ["A", "A", "A", "B", "B", "B"],
        }
    )

    metrics = MulticlassClassificationMetrics(
        y_true="target", y_preds="pred", by="group"
    )
    result = metrics.transform(df)

    assert result.columns == ["by", "prediction", "metric", "value"]
    groups = result["by"].unique().to_list()
    assert "A" in groups
    assert "B" in groups


def test_multiclass_classification_metrics_selector():
    df = pl.DataFrame(
        {
            "target": [0, 1, 2, 0, 1, 2],
            "pred1": [0, 1, 2, 0, 1, 2],
            "pred2": [0, 1, 2, 0, 1, 2],
            "other": [0, 1, 2, 0, 1, 2],
        }
    )

    metrics = MulticlassClassificationMetrics(
        y_true="target", y_preds=cs.starts_with("pred")
    )
    result = metrics.transform(df)

    predictions = result["prediction"].unique().to_list()
    assert "pred1" in predictions
    assert "pred2" in predictions
    assert "other" not in predictions


def test_multiclass_classification_metrics_missing_y_true():
    df = pl.DataFrame(
        {
            "pred": [0, 1, 2],
        }
    )

    metrics = MulticlassClassificationMetrics(y_true="target", y_preds="pred")
    with pytest.raises(ValueError, match="y_true column 'target' not found in data"):
        metrics.transform(df)


def test_multiclass_classification_metrics_zero_division():
    # Test case where some classes are missing in pred or true
    df = pl.DataFrame({"target": [0, 1, 2], "pred": [0, 0, 0]})

    metrics = MulticlassClassificationMetrics(y_true="target", y_preds="pred")
    result = metrics.transform(df)

    # Should not raise error and return 1/9 for precision_macro
    # Class 0: 1/3, Class 1: 0, Class 2: 0 -> Macro: (1/3 + 0 + 0) / 3 = 1/9
    precision = result.filter(pl.col("metric") == "precision_macro")["value"][0]
    assert pytest.approx(precision) == 1 / 9
