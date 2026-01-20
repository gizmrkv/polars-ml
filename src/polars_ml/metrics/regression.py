from __future__ import annotations

from typing import Any, Iterable

from numpy.typing import NDArray
from polars import DataFrame
from polars._typing import ColumnNameOrSelector
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
    root_mean_squared_error,
    root_mean_squared_log_error,
)

from polars_ml.base import Transformer


class RegressionMetrics(Transformer):
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
        mse = mean_squared_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)

        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        r2 = r2_score(y_true, y_pred)

        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2,
        }

        if (y_true >= 0).all() and (y_pred >= 0).all():
            msle = mean_squared_log_error(y_true, y_pred)
            rmsle = root_mean_squared_log_error(y_true, y_pred)
            metrics["msle"] = msle
            metrics["rmsle"] = rmsle

        return metrics
