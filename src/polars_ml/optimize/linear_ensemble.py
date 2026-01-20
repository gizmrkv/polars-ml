from __future__ import annotations

from typing import Iterable, Self

import numpy as np
import polars as pl
from polars import DataFrame
from polars._typing import ColumnNameOrSelector
from sklearn.linear_model import ElasticNet

from polars_ml.base import Transformer
from polars_ml.exceptions import NotFittedError


class LinearEnsemble(Transformer):
    def __init__(
        self,
        pred_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        target_column: str,
        *,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = False,
        positive: bool = True,
        max_iter: int = 1000,
        output_column: str = "ensemble",
    ):
        self.pred_columns = pred_columns
        self.target_column = target_column
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.positive = positive
        self.max_iter = max_iter
        self.output_column = output_column

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
        X_list = []
        y_list = []
        for dataset in more_data.values():
            X = dataset.select(pred_columns).to_numpy()
            y = dataset[self.target_column].to_numpy()
            X_list.append(X)
            y_list.append(y)

        # Concatenate all validation data
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)

        # Train ElasticNet model
        self.model_ = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            fit_intercept=self.fit_intercept,
            positive=self.positive,
            max_iter=self.max_iter,
        )
        self.model_.fit(X_train, y_train)

        self.pred_columns_ = pred_columns
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if not hasattr(self, "model_"):
            raise NotFittedError()

        X = data.select(self.pred_columns_).to_numpy()
        y_pred = self.model_.predict(X)

        return data.with_columns(pl.Series(self.output_column, y_pred))

    def get_weights(self) -> dict[str, float]:
        if not hasattr(self, "model_"):
            raise ValueError("The transformer has not been fitted yet.")
        return dict(zip(self.pred_columns_, self.model_.coef_))

    def get_intercept(self) -> float:
        if not hasattr(self, "model_"):
            raise ValueError("The transformer has not been fitted yet.")
        return float(self.model_.intercept_)
