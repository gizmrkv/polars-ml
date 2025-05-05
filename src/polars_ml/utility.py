import itertools
import uuid
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import polars as pl
from numpy.typing import NDArray
from polars import DataFrame, Expr, Series
from polars._typing import (
    ColumnNameOrSelector,
    IntoExpr,
)

from .component import Component


class iter_axes:
    def __init__(
        self,
        data: DataFrame,
        *axes: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
    ):
        self.columns = [
            cols
            for cols in list(
                itertools.product(
                    *[data.lazy().select(ax).collect_schema().names() for ax in axes]
                )
            )
            if len(set(cols)) == len(cols)
        ]

    def __iter__(self) -> Iterator[tuple[str, ...]]:
        return iter(self.columns)

    def __len__(self) -> int:
        return len(self.columns)


class Apply(Component):
    def __init__(self, func: Callable[[DataFrame], DataFrame | Any]):
        self.func = func

    def transform(self, data: DataFrame) -> DataFrame:
        output = self.func(data)
        if isinstance(output, DataFrame):
            return output
        else:
            return data


class GroupByThen(Component):
    def __init__(
        self,
        by: str | Expr | Sequence[str | Expr] | None = None,
        *aggs: IntoExpr | Iterable[IntoExpr],
        maintain_order: bool = False,
        after_with_columns: IntoExpr | Iterable[IntoExpr] | None = None,
    ):
        self.by = by
        self.aggs = aggs
        self.maintain_order = maintain_order
        self.after_with_columns = after_with_columns

    def fit(
        self,
        data: pl.DataFrame,
        validation_data: pl.DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> "GroupByThen":
        self.grouped = data.group_by(self.by, maintain_order=self.maintain_order).agg(
            *self.aggs
        )
        if self.after_with_columns is not None:
            self.grouped = self.grouped.with_columns(self.after_with_columns)

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        return data.join(self.grouped, on=self.by, how="left")


class Impute(Component):
    def __init__(
        self, imputer: Component, column: str, *, maintain_order: bool = False
    ):
        self.imputer = imputer
        self.column = column
        self.maintain_order = maintain_order

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> "Impute":
        train_data = data.filter(pl.col(self.column).is_not_null())

        if isinstance(validation_data, DataFrame):
            validation_data = validation_data.filter(pl.col(self.column).is_not_null())
        elif isinstance(validation_data, Mapping):
            validation_data = {
                key: value.filter(pl.col(self.column).is_not_null())
                for key, value in validation_data.items()
            }

        self.imputer.fit(train_data, validation_data)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if self.maintain_order:
            index_name = uuid.uuid4().hex
            data = data.with_row_index(index_name)
            missing_data = data.filter(pl.col(self.column).is_null())
            imputed_data = self.imputer.transform(missing_data.drop(index_name))
            filled_data = missing_data.with_columns(imputed_data[self.column])
            data = pl.concat(
                [data.filter(pl.col(self.column).is_not_null()), filled_data]
            )
            return data.sort(index_name).drop(index_name)
        else:
            missing_data = data.filter(pl.col(self.column).is_null())
            imputed_data = self.imputer.transform(missing_data)
            filled_data = missing_data.with_columns(imputed_data[self.column])
            data = pl.concat(
                [data.filter(pl.col(self.column).is_not_null()), filled_data]
            )
            return data


class ScikitLearnWrapper(Component):
    def __init__(
        self,
        model: Any,
        *,
        label: str | Iterable[str] | None = None,
        exclude: str | Iterable[str] | None = None,
        fit_params: Mapping[str, Any] | None = None,
        predict_method: str = "predict",
        predict_params: Mapping[str, Any] | None = None,
        prediction_name: str = "prediction",
        include_input: bool = True,
    ):
        self.model = model
        self.labels = list(label) if label else []

        if isinstance(exclude, str):
            self.exclude = [exclude]
        elif isinstance(exclude, Iterable):
            self.exclude = list(exclude)
        else:
            self.exclude = []

        self.prediction_name = prediction_name
        self.include_input = include_input
        self.fit_params = fit_params or {}
        self.predict_method = predict_method
        self.predict_params = predict_params or {}

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> "ScikitLearnWrapper":
        train_features = data.select(pl.exclude(*self.labels, *self.exclude))
        train_labels = data.select(*self.labels)
        self.feature_names = train_features.columns

        train_X = train_features.to_numpy().squeeze()
        train_y = train_labels.to_numpy().squeeze()
        self.model.fit(train_X, train_y, **self.fit_params)

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input = data.select(self.feature_names).to_numpy().squeeze()
        pred: NDArray[Any] = getattr(self.model, self.predict_method)(
            input, **self.predict_params
        )

        if pred.ndim == 1:
            columns = [Series(self.prediction_name, pred)]
        else:
            n = pred.shape[1]
            zero_pad = len(str(n))
            columns = [
                Series(f"{self.prediction_name}_{i:0{zero_pad}d}", pred[:, i])
                for i in range(n)
            ]

        if self.include_input:
            return data.with_columns(columns)
        else:
            return DataFrame(columns)
