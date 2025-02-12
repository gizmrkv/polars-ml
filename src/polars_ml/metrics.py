from typing import Any, Callable

from polars import DataFrame, Series
from sklearn import metrics


def evaluate_metrics(
    data: DataFrame,
    y_true: str,
    y_pred: str,
    *,
    metrics: dict[str, Callable[[Series, Series], Any]],
    by: str | None = None,
) -> DataFrame:
    return DataFrame(
        [
            {
                "level": by,
                **{name: metric(y_true, y_pred) for name, metric in metrics.items()},
            }
            for by, y_true, y_pred in (
                data.group_by(by).agg(y_true, y_pred).sort(by).iter_rows()
            )
        ]
    )


def evaluate_regression_metrics(
    data: DataFrame, y_true: str, y_pred: str, *, by: str | None = None
) -> DataFrame:
    return evaluate_metrics(
        data,
        y_true,
        y_pred,
        metrics={
            "mse": metrics.mean_squared_error,
            "rmse": metrics.root_mean_squared_error,
            "mae": metrics.mean_absolute_error,
            "r2": metrics.r2_score,
        },
        by=by,
    )


def evaluate_classification_metrics(
    data: DataFrame, y_true: str, y_pred: str, *, by: str | None = None
) -> DataFrame:
    return evaluate_metrics(
        data,
        y_true,
        y_pred,
        metrics={
            "accuracy": metrics.accuracy_score,
            "precision": metrics.precision_score,
            "recall": metrics.recall_score,
            "f1": metrics.f1_score,
        },
        by=by,
    )


def evaluate_binary_classification_metrics(
    data: DataFrame, y_true: str, y_pred: str, *, by: str | None = None
) -> DataFrame:
    return evaluate_metrics(
        data,
        y_true,
        y_pred,
        metrics={
            "roc_auc": metrics.roc_auc_score,
            "average_precision": metrics.average_precision_score,
            "brier": metrics.brier_score_loss,
            "log_loss": metrics.log_loss,
        },
        by=by,
    )
