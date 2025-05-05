import uuid
from typing import Callable, Iterable, Mapping

import polars as pl
from polars import DataFrame, Series
from polars._typing import IntoExpr

from polars_ml.component import Component
from polars_ml.model_selection import KFold


class Stacker(Component):
    def __init__(
        self,
        model_fn: Callable[[DataFrame, int], Component],
        k_fold: KFold,
        *,
        aggs_on_transform: IntoExpr | Iterable[IntoExpr] = pl.all().mean(),
    ):
        self.model_fn = model_fn
        self.k_fold = k_fold
        self.aggs_on_transform = aggs_on_transform
        self.index_name = uuid.uuid4().hex

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> "Stacker":
        if isinstance(validation_data, DataFrame):
            validation_data = {"validation": validation_data}
        if isinstance(validation_data, Mapping):
            validation_data = dict(**validation_data)

        self.models: list[Component] = []
        self.valid_indexes: list[Series] = []
        for i, (train_idx, valid_idx) in enumerate(self.k_fold.split(data)):
            train_data = data.select(pl.all().gather(train_idx))
            valid_data = data.select(pl.all().gather(valid_idx))

            if validation_data is not None:
                valid_data = {"validation_fold": valid_data} | validation_data

            model = self.model_fn(train_data, i)
            model.fit(train_data, valid_data)

            self.models.append(model)

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        return (
            pl.concat(
                [
                    model.transform(data).with_row_index(self.index_name)
                    for model in self.models
                ]
            )
            .group_by(self.index_name)
            .agg(self.aggs_on_transform)
            .sort(self.index_name)
            .drop(self.index_name)
        )

    def fit_transform(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> DataFrame:
        self.fit(data, validation_data)
        return (
            pl.concat(
                [
                    model.transform(
                        data.select(pl.all().gather(valid_idx))
                    ).with_columns(valid_idx)
                    for model, valid_idx in zip(self.models, self.valid_indexes)
                ]
            )
            .sort(self.index_name)
            .drop(self.index_name)
        )
