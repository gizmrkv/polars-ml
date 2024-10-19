import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, List, Self, override

import polars as pl
from polars import DataFrame, LazyFrame
from polars._typing import IntoExpr

from polars_ml.component import Component, ComponentList, LazyComponent
from polars_ml.exception import NotFittedError

from .train_valid import TrainValid


class Stacker(LazyComponent):
    def __init__(
        self,
        model_fn: Callable[[], Component | LazyComponent],
        *,
        on_train_fn: Callable[[], Component | LazyComponent] | None = None,
        on_valid_fn: Callable[[], Component | LazyComponent] | None = None,
        aggs: IntoExpr | Iterable[IntoExpr] = pl.all().mean(),
        fold_name: str = "fold",
    ):
        self.model_fn = model_fn
        self.on_train_fn = on_train_fn
        self.on_valid_fn = on_valid_fn
        self.aggs = aggs
        self.fold_name = fold_name

        self.index_name = uuid.uuid4().hex
        self.is_valid_column = uuid.uuid4().hex
        self.train_valid_models: ComponentList[LazyComponent] | None = None
        self.folds: List[Any] = []
        self.valid_indexes: List[DataFrame] = []

    @override
    def is_fitted(self) -> bool:
        return (
            self.train_valid_models is not None and self.train_valid_models.is_fitted()
        )

    @override
    def set_log_dir(self, log_dir: str | Path | None) -> Self:
        if self.train_valid_models:
            self.train_valid_models.set_log_dir(log_dir)
        return self

    @override
    def fit(self, data: LazyFrame) -> Self:
        self.fitted = True
        fold_df = data.select(self.fold_name).unique().collect()
        self.folds = fold_df[self.fold_name].to_list()
        self.train_valid_models = ComponentList(
            [
                TrainValid(
                    self.model_fn(),
                    on_train=self.on_train_fn() if self.on_train_fn else None,
                    on_valid=self.on_valid_fn() if self.on_valid_fn else None,
                    is_valid_column=self.is_valid_column,
                )
                for _ in self.folds
            ]
        )
        self.train_valid_models.set_log_dir(self.log_dir)

        self.valid_indexes.clear()
        for fold, train_valid_model in zip(self.folds, self.train_valid_models):
            train_data = (
                data.with_row_index(self.index_name)
                .with_columns(
                    pl.col(self.fold_name).eq(fold).alias(self.is_valid_column)
                )
                .drop(self.fold_name)
            )
            valid_index = (
                train_data.select(self.is_valid_column, self.index_name)
                .filter(self.is_valid_column)
                .select(self.index_name)
            )

            train_valid_model.fit(train_data.drop(self.index_name))
            self.valid_indexes.append(valid_index.collect())

        return self

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        if self.train_valid_models is None:
            raise NotFittedError()

        preds = [model.execute(data) for model in self.train_valid_models]
        pred_catted = pl.concat(
            [pred.with_row_index(self.index_name) for pred in preds]
        )
        pred = (
            pred_catted.group_by(self.index_name)
            .agg(self.aggs)
            .sort(self.index_name)
            .drop(self.index_name)
        )
        return pred

    @override
    def fit_execute(self, data: LazyFrame) -> LazyFrame:
        self.fit(data)
        assert self.train_valid_models is not None

        valid_index = pl.concat(self.valid_indexes).lazy()
        preds = pl.concat(
            [
                model.execute(data.filter(pl.col(self.fold_name) == fold))
                for fold, model in zip(self.folds, self.train_valid_models)
            ]
        )
        pred = (
            pl.concat([preds, valid_index], how="horizontal")
            .sort(self.index_name)
            .drop(self.index_name)
        )
        return pred
