from __future__ import annotations

from typing import Any, Iterable, Self

import numpy as np
from numpy.typing import NDArray
from polars import DataFrame
from polars._typing import ColumnNameOrSelector
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)

from polars_ml.base import Transformer
from polars_ml.exceptions import NotFittedError


class BinaryClassificationMetrics(Transformer):
    def __init__(
        self,
        y_true: str,
        y_preds: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *,
        by: str | None = None,
    ):
        self.y_true = y_true
        self.y_preds = y_preds
        self.by = by

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self.n_train_sample = len(data)
        self.n_train_positive = data[self.y_true].sum()
        self.n_train_negative = self.n_train_sample - self.n_train_positive
        self.train_pos_rate = (
            self.n_train_positive / self.n_train_sample
            if self.n_train_sample > 0
            else 1e-15
        )
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if not hasattr(self, "train_pos_rate"):
            raise NotFittedError()

        if self.y_true not in data.columns:
            raise ValueError(f"y_true column '{self.y_true}' not found in data")

        metrics_list = []
        if self.by is None:
            y_pred_cols = data.select(self.y_preds).columns
            for y_pred_col in y_pred_cols:
                y_true = data[self.y_true].to_numpy()
                y_pred = data[y_pred_col].to_numpy()

                n_sample = len(y_true)
                n_positive = y_true.sum()
                if n_positive == 0 or n_positive == n_sample:
                    continue

                metrics = self.calc_metrics(y_true, y_pred)
                metrics_list.extend(
                    [
                        {"prediction": y_pred_col, "metric": k, "value": v}
                        for k, v in metrics.items()
                    ]
                )
            metrics_df = DataFrame(metrics_list).select("prediction", "metric", "value")

        else:
            for (by,), group in data.partition_by(
                self.by, as_dict=True, maintain_order=True
            ).items():
                y_pred_cols = group.select(self.y_preds).columns
                for y_pred_col in y_pred_cols:
                    y_true = group[self.y_true].to_numpy()
                    y_pred = group[y_pred_col].to_numpy()

                    n_sample = len(y_true)
                    n_positive = y_true.sum()
                    if n_positive == 0 or n_positive == n_sample:
                        continue

                    metrics = self.calc_metrics(y_true, y_pred)
                    metrics_list.extend(
                        [
                            {
                                "by": by,
                                "prediction": y_pred_col,
                                "metric": k,
                                "value": v,
                            }
                            for k, v in metrics.items()
                        ]
                    )

            metrics_df = DataFrame(metrics_list).select(
                "by", "prediction", "metric", "value"
            )

        return metrics_df

    def calc_metrics(
        self, y_true: NDArray[Any], y_pred: NDArray[Any]
    ) -> dict[str, Any]:
        n_sample = len(y_true)
        n_positive = y_true.sum()
        pos_rate = n_positive / n_sample if n_sample > 0 else 0.0

        log_loss_value = log_loss(y_true, y_pred)
        entropy = -(
            pos_rate * np.log(self.train_pos_rate)
            + (1 - pos_rate) * np.log(1 - self.train_pos_rate)
        )
        normalized_entropy = log_loss_value / entropy if entropy > 0 else 0.0

        roc_auc = roc_auc_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        pr_auc = auc(recall, precision)
        average_precision = average_precision_score(y_true, y_pred)
        brier_score = brier_score_loss(y_true, y_pred)

        return {
            "n_sample": n_sample,
            "n_positive": n_positive,
            "pos_rate": pos_rate,
            "log_loss": log_loss_value,
            "normalized_entropy": normalized_entropy,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "average_precision": average_precision,
            "brier_score": brier_score,
        }
