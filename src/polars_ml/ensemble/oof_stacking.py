from __future__ import annotations

import uuid
from typing import Callable, Iterable, Mapping, Self

import polars as pl
from polars import DataFrame, Series
from polars._typing import IntoExpr

from polars_ml.base import Transformer
from polars_ml.model_selection import KFold


class OOFStacking(Transformer):
    def __init__(
        self,
        model_fn: Callable[[DataFrame, int], Transformer],
        k_fold: KFold,
        *,
        aggs_on_transform: IntoExpr | Iterable[IntoExpr] = pl.all().mean(),
    ):
        self.model_fn = model_fn
        self.k_fold = k_fold
        self.aggs_on_transform = aggs_on_transform
        self._index_name = f"__index_{uuid.uuid4().hex[:8]}"
        self.models: list[Transformer] = []
        self.valid_indexes: list[Series] = []

    def fit(
        self,
        data: DataFrame,
        **more_data: DataFrame,
    ) -> Self:
        self.models = []
        self.valid_indexes = []

        for i, (train_idx, valid_idx) in enumerate(self.k_fold.split(data)):
            train_data = data.select(pl.all().gather(train_idx))
            valid_data_fold = data.select(pl.all().gather(valid_idx))

            validation_sets = {"validation_fold": valid_data_fold}
            validation_sets.update(more_data)

            model = self.model_fn(train_data, i)
            model.fit(train_data, **validation_sets)

            self.models.append(model)
            self.valid_indexes.append(valid_idx)

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        return (
            pl.concat(
                [
                    model.transform(data).with_columns(
                        pl.Series(self._index_name, range(len(data)))
                    )
                    for model in self.models
                ]
            )
            .group_by(self._index_name)
            .agg(self.aggs_on_transform)
            .sort(self._index_name)
            .drop(self._index_name)
        )

    def fit_transform(
        self,
        data: DataFrame,
        **more_data: DataFrame,
    ) -> DataFrame:
        self.fit(data, **more_data)

        oof_predictions = []
        for model, valid_idx in zip(self.models, self.valid_indexes):
            valid_data = data.select(pl.all().gather(valid_idx))
            pred = model.transform(valid_data).with_columns(
                valid_idx.alias(self._index_name)
            )
            oof_predictions.append(pred)

        return pl.concat(oof_predictions).sort(self._index_name).drop(self._index_name)
