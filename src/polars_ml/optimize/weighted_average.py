from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Self

import numpy as np
import polars as pl
from polars import DataFrame
from polars._typing import ColumnNameOrSelector
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

from polars_ml.base import Transformer
from polars_ml.exceptions import NotFittedError


class WeightedAverage(Transformer):
    def __init__(
        self,
        pred_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        target_column: str,
        *,
        metric_fn: Callable[[NDArray[Any], NDArray[Any]], float] | None = None,
        is_higher_better: bool = False,
        method: str = "SLSQP",
        sum_to_one: bool = True,
        non_negative: bool = True,
        output_column: str = "weighted_average",
        scipy_kwargs: Mapping[str, Any] | None = None,
    ):
        self.pred_columns = pred_columns
        self.target_column = target_column
        self.metric_fn = metric_fn or mean_squared_error
        self.is_higher_better = is_higher_better
        self.method = method
        self.sum_to_one = sum_to_one
        self.non_negative = non_negative
        self.output_column = output_column
        self.scipy_kwargs = scipy_kwargs or {}

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        if not more_data:
            raise ValueError(
                "more_data must contain at least one dataset for optimization."
            )

        pred_columns = data.select(self.pred_columns).columns
        n_preds = len(pred_columns)

        if n_preds == 0:
            raise ValueError("No prediction columns selected.")

        # Collect all validation datasets
        validation_datasets = []
        for dataset in more_data.values():
            y_true = dataset[self.target_column].to_numpy()
            y_preds = dataset.select(pred_columns).to_numpy()
            validation_datasets.append((y_true, y_preds))

        def objective(weights: NDArray[Any]) -> float:
            # Calculate average metric across all validation datasets
            scores = []
            for y_true, y_preds in validation_datasets:
                y_avg = np.dot(y_preds, weights)
                score = self.metric_fn(y_true, y_avg)
                scores.append(score)

            avg_score = np.mean(scores)
            return -avg_score if self.is_higher_better else avg_score

        bounds = None
        if self.non_negative:
            bounds = [(0, None) for _ in range(n_preds)]

        constraints = []
        if self.sum_to_one:
            constraints.append({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})

        init_weights = np.ones(n_preds) / n_preds

        res = minimize(
            objective,
            init_weights,
            method=self.method,
            bounds=bounds,
            constraints=constraints,
            **self.scipy_kwargs,
        )

        if not res.success:
            # We don't necessarily want to raise an error, but maybe log a warning?
            # For now, let's just proceed with whatever we got.
            pass

        self.weights_ = res.x
        self.pred_columns_ = pred_columns
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if not hasattr(self, "weights_"):
            raise NotFittedError()

        # Calculate weighted average
        # We use pl.sum_horizontal with weights by scaling columns first
        exprs = [
            pl.col(col) * weight
            for col, weight in zip(self.pred_columns_, self.weights_)
        ]
        return data.with_columns(pl.sum_horizontal(exprs).alias(self.output_column))

    def get_weights(self) -> dict[str, float]:
        if not hasattr(self, "weights_"):
            raise ValueError("The transformer has not been fitted yet.")
        return dict(zip(self.pred_columns_, self.weights_))
