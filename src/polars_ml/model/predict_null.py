import uuid
from pathlib import Path
from typing import Self, override

import polars as pl
from polars import LazyFrame

from polars_ml.component import Component, LazyComponent
from polars_ml.utils import Lazy


class PredictNull(LazyComponent):
    def __init__(self, target: str, model: Component | LazyComponent):
        self.target = target
        self.model = Lazy(model) if isinstance(model, Component) else model

    @override
    def is_fitted(self) -> bool:
        return self.model.is_fitted()

    @override
    def set_log_dir(self, log_dir: str | Path | None) -> Self:
        self.model.set_log_dir(log_dir)
        return self

    @override
    def fit(self, data: LazyFrame) -> Self:
        data_fill = data.filter(pl.col(self.target).is_not_null())
        self.model.fit(data_fill)
        return self

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        # Add index to restore the order later
        index_name = str(uuid.uuid4())
        data = data.with_row_index(index_name)

        # Predict rows with missing target feature
        X_null = data.filter(pl.col(self.target).is_null()).drop(self.target)
        y_pred = self.model.execute(X_null.drop(index_name))

        # Fill the missing values
        X_fill = data.filter(pl.col(self.target).is_not_null())
        X_pred = pl.concat([X_null, y_pred], how="horizontal")
        X_filled = (
            pl.concat([X_pred.select(X_fill.columns), X_fill])
            .sort(index_name)
            .drop(index_name)
        )
        return X_filled
