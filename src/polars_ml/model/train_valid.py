from pathlib import Path
from typing import Self, override

import polars as pl
from polars import LazyFrame

from polars_ml.component import Component, ComponentDict, LazyComponent
from polars_ml.utils import Lazy


class TrainValid(LazyComponent):
    def __init__(
        self,
        model: Component | LazyComponent,
        *,
        on_train: Component | LazyComponent | None = None,
        on_valid: Component | LazyComponent | None = None,
        is_valid_column: str = "is_valid",
    ):
        self.model = Lazy(model) if isinstance(model, Component) else model
        self.on_train = Lazy(on_train) if isinstance(on_train, Component) else on_train
        self.on_valid = Lazy(on_valid) if isinstance(on_valid, Component) else on_valid
        self.is_valid_column = is_valid_column

        self.components = ComponentDict({"model": self.model})
        if self.on_train:
            self.components["on_train"] = self.on_train
        if self.on_valid:
            self.components["on_valid"] = self.on_valid

    @override
    def is_fitted(self) -> bool:
        return self.components.is_fitted()

    @override
    def set_log_dir(self, log_dir: str | Path | None) -> Self:
        self.components.set_log_dir(log_dir)
        return self

    @override
    def fit(self, data: LazyFrame) -> Self:
        self.fit_execute(data)
        return self

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        model_output = self.model.execute(data)

        if self.on_train:
            train_output = model_output.filter(pl.col(self.is_valid_column).not_())
            self.on_train.execute(train_output)
        if self.on_valid:
            valid_output = model_output.filter(pl.col(self.is_valid_column))
            self.on_valid.execute(valid_output)

        return model_output

    @override
    def fit_execute(self, data: LazyFrame) -> LazyFrame:
        model_output = self.model.fit_execute(data)

        if self.on_train:
            train_output = model_output.filter(pl.col(self.is_valid_column).not_())
            self.on_train.fit_execute(train_output)
        if self.on_valid:
            valid_output = model_output.filter(pl.col(self.is_valid_column))
            self.on_valid.fit_execute(valid_output)

        return model_output
