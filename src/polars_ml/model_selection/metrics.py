import uuid
from typing import Any

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.typing import NDArray
from polars import DataFrame
from scipy import optimize
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    explained_variance_score,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def evaluate_regression_metrics(
    data: DataFrame, y_true: str, y_pred: str, *, by: str | None = None
) -> DataFrame:
    if by is None:
        return _evaluate_regression_metrics(data, y_true, y_pred)
    else:
        groups = data[by].unique(maintain_order=True)
        metrics = []
        for group in groups:
            group_data = data.filter(pl.col(by) == group)
            metrics.append(
                _evaluate_regression_metrics(group_data, y_true, y_pred).with_columns(
                    pl.lit(group).alias(by)
                )
            )

        return pl.concat(metrics)


def _evaluate_regression_metrics(
    data: DataFrame, y_true: str, y_pred: str
) -> DataFrame:
    y_true_values = data[y_true].to_numpy()
    y_pred_values = data[y_pred].to_numpy()

    metrics = {
        "mse": mean_squared_error(y_true_values, y_pred_values),
        "rmse": np.sqrt(mean_squared_error(y_true_values, y_pred_values)),
        "mae": mean_absolute_error(y_true_values, y_pred_values),
        "r2": r2_score(y_true_values, y_pred_values),
        "explained_variance": explained_variance_score(y_true_values, y_pred_values),
        "max_error": max_error(y_true_values, y_pred_values),
        "median_absolute_error": median_absolute_error(y_true_values, y_pred_values),
    }

    metrics["mape"] = mean_absolute_percentage_error(y_true_values, y_pred_values)

    if np.all(y_true_values > 0) and np.all(y_pred_values > 0):
        metrics["msle"] = mean_squared_log_error(y_true_values, y_pred_values)
        metrics["rmsle"] = np.sqrt(mean_squared_log_error(y_true_values, y_pred_values))

    metrics["pearson_corr"] = np.corrcoef(y_true_values, y_pred_values)[0, 1]

    return DataFrame(metrics)


def evaluate_classification_metrics(
    data: DataFrame,
    y_true: str,
    y_pred_class: str | None = None,
    y_pred_proba_prefix: str | None = None,
    *,
    by: str | None = None,
    n_classes: int | None = None,
) -> DataFrame:
    if by is None:
        return _evaluate_classification_metrics(
            data,
            y_true,
            y_pred_class=y_pred_class,
            y_pred_proba_prefix=y_pred_proba_prefix,
            n_classes=n_classes,
        )
    else:
        groups = data[by].unique(maintain_order=True)
        metrics = []
        for group in groups:
            group_data = data.filter(pl.col(by) == group)
            metrics.append(
                _evaluate_classification_metrics(
                    group_data,
                    y_true,
                    y_pred_class=y_pred_class,
                    y_pred_proba_prefix=y_pred_proba_prefix,
                    n_classes=n_classes,
                ).with_columns(pl.lit(group).alias(by))
            )

        return pl.concat(metrics)


def _evaluate_classification_metrics(
    data: DataFrame,
    y_true: str,
    y_pred_class: str | None = None,
    y_pred_proba_prefix: str | None = None,
    *,
    n_classes: int | None = None,
) -> DataFrame:
    if y_pred_class is None and y_pred_proba_prefix is None:
        raise ValueError(
            "At least one of y_pred_class and y_pred_proba_prefix is required"
        )

    if y_pred_class is None and y_pred_proba_prefix is not None:
        import polars_ml as pml

        y_pred_class = uuid.uuid4().hex
        data = pml.Pipeline(
            pml.HorizontalArgMax(
                cs.starts_with(y_pred_proba_prefix), value_name=y_pred_class
            ),
            lambda df: df.with_columns(
                pl.col(y_pred_class)
                .list.first()
                .str.extract(rf"{y_pred_proba_prefix}(\d+)")
                .cast(pl.UInt8)
            ),
        ).transform(data)

    assert isinstance(y_pred_class, str)

    labels = list(range(n_classes)) if n_classes is not None else None
    y_true_values = data[y_true].to_numpy()
    y_pred_values = data[y_pred_class].to_numpy()
    metrics = {
        "accuracy": accuracy_score(y_true_values, y_pred_values),
        "balanced_accuracy": balanced_accuracy_score(y_true_values, y_pred_values),
        "precision_macro": precision_score(
            y_true_values, y_pred_values, average="macro", labels=labels
        ),
        "precision_weighted": precision_score(
            y_true_values, y_pred_values, average="weighted", labels=labels
        ),
        "recall_macro": recall_score(
            y_true_values, y_pred_values, average="macro", labels=labels
        ),
        "recall_weighted": recall_score(
            y_true_values, y_pred_values, average="weighted", labels=labels
        ),
        "f1_macro": f1_score(
            y_true_values, y_pred_values, average="macro", labels=labels
        ),
        "f1_weighted": f1_score(
            y_true_values, y_pred_values, average="weighted", labels=labels
        ),
        "matthews_corrcoef": matthews_corrcoef(y_true_values, y_pred_values),
        "cohen_kappa_score": cohen_kappa_score(y_true_values, y_pred_values),
    }

    if y_pred_proba_prefix is not None:
        y_pred_proba_values = data.select(
            cs.starts_with(y_pred_proba_prefix)
        ).to_numpy()
        metrics |= {
            "roc_auc_ovo": roc_auc_score(
                y_true_values, y_pred_proba_values, multi_class="ovo", labels=labels
            ),
            "log_loss": log_loss(y_true_values, y_pred_proba_values, labels=labels),
        }

    return DataFrame(metrics)


def evaluate_binary_classification_metrics(
    data: DataFrame,
    y_true: str,
    y_pred_class: str | None = None,
    y_pred_proba: str | None = None,
    *,
    by: str | None = None,
    pos_label: int = 1,
    threshold: float = 0.5,
) -> DataFrame:
    if by is None:
        return _evaluate_binary_classification_metrics(
            data,
            y_true,
            y_pred_class=y_pred_class,
            y_pred_proba=y_pred_proba,
            pos_label=pos_label,
            threshold=threshold,
        )
    else:
        groups = data[by].unique(maintain_order=True)
        metrics = []
        for group in groups:
            group_data = data.filter(pl.col(by) == group)
            metrics.append(
                _evaluate_binary_classification_metrics(
                    group_data,
                    y_true,
                    y_pred_class=y_pred_class,
                    y_pred_proba=y_pred_proba,
                    pos_label=pos_label,
                    threshold=threshold,
                ).with_columns(pl.lit(group).alias(by))
            )

        return pl.concat(metrics)


def _evaluate_binary_classification_metrics(
    data: DataFrame,
    y_true: str,
    y_pred_class: str | None = None,
    y_pred_proba: str | None = None,
    *,
    pos_label: int = 1,
    threshold: float = 0.5,
    beta: float = 1.0,
) -> DataFrame:
    if y_pred_class is None and y_pred_proba is None:
        raise ValueError("At least one of y_pred_class and y_pred_proba is required")

    if y_pred_class is None and y_pred_proba is not None:
        y_pred_class = uuid.uuid4().hex
        data = data.with_columns(
            (pl.col(y_pred_proba) >= threshold).cast(pl.Int8).alias(y_pred_class)
        )

    if y_pred_proba is None and y_pred_class is None:
        raise ValueError(
            "y_pred_proba or y_pred_class is required to calculate metrics"
        )

    assert isinstance(y_pred_class, str)

    y_true_values = data[y_true].to_numpy()
    y_pred_values = data[y_pred_class].to_numpy()

    metrics = {
        "accuracy": accuracy_score(y_true_values, y_pred_values),
        "balanced_accuracy": balanced_accuracy_score(y_true_values, y_pred_values),
        "precision": precision_score(y_true_values, y_pred_values, pos_label=pos_label),
        "recall": recall_score(y_true_values, y_pred_values, pos_label=pos_label),
        "f1": f1_score(y_true_values, y_pred_values, pos_label=pos_label),
        "matthews_corrcoef": matthews_corrcoef(y_true_values, y_pred_values),
        "cohen_kappa": cohen_kappa_score(y_true_values, y_pred_values),
    }

    if y_pred_proba is not None:
        y_pred_proba_values = data[y_pred_proba].to_numpy()
        precision, recall, _ = precision_recall_curve(y_true_values, y_pred_values)
        metrics |= {
            "log_loss": log_loss(y_true_values, y_pred_proba_values),
            "brier_score": brier_score_loss(y_true_values, y_pred_proba_values),
            "roc_auc": roc_auc_score(y_true_values, y_pred_proba_values),
            "pr_auc": auc(recall, precision),
            "average_precision": average_precision_score(
                y_true_values, y_pred_proba_values
            ),
        }

        def negative_fbeta_score(
            threshold: float,
            y_true: NDArray[Any],
            y_pred_proba: NDArray[Any],
            beta: float = beta,
        ) -> float:
            y_pred = (y_pred_proba >= threshold).astype(int)
            return -float(fbeta_score(y_true, y_pred, beta=beta, pos_label=pos_label))

        result = optimize.minimize_scalar(
            negative_fbeta_score,
            bounds=(0, 1),
            method="bounded",
            args=(y_true_values, y_pred_proba_values, beta),
        )

        optimal_threshold = result.x  # type: ignore
        optimal_predictions = (y_pred_proba_values >= optimal_threshold).astype(int)
        metrics |= {
            "optimal_threshold": optimal_threshold,
            "optimal_fbeta": fbeta_score(
                y_true_values, optimal_predictions, beta=beta, pos_label=pos_label
            ),
            "optimal_precision": precision_score(
                y_true_values, optimal_predictions, pos_label=pos_label
            ),
            "optimal_recall": recall_score(
                y_true_values, optimal_predictions, pos_label=pos_label
            ),
            "optimal_f1": f1_score(
                y_true_values, optimal_predictions, pos_label=pos_label
            ),
            "optimal_accuracy": accuracy_score(y_true_values, optimal_predictions),
            "optimal_balanced_accuracy": balanced_accuracy_score(
                y_true_values, optimal_predictions
            ),
        }

    return DataFrame(metrics)
