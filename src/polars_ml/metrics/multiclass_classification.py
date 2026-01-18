from __future__ import annotations

from typing import Any, Iterable

from numpy.typing import NDArray
from polars import DataFrame
from polars._typing import ColumnNameOrSelector
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from polars_ml.base import Transformer


class MulticlassClassificationMetrics(Transformer):
    def __init__(
        self,
        y_true: str,
        y_preds: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *,
        by: str | None = None,
    ) -> None:
        self.y_true = y_true
        self.y_preds = y_preds
        self.by = by

    def transform(self, data: DataFrame) -> DataFrame:
        if self.y_true not in data.columns:
            raise ValueError(f"y_true column '{self.y_true}' not found in data")

        metrics_list = []
        if self.by is None:
            y_pred_cols = data.select(self.y_preds).columns
            for y_pred_col in y_pred_cols:
                y_true = data[self.y_true].to_numpy()
                y_pred = data[y_pred_col].to_numpy()

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
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        return {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
        }
