from typing import Iterable, Literal, Mapping, Self, Sequence

import polars as pl
from polars import DataFrame, Expr
from polars._typing import ConcatMethod, IntoExpr

from polars_ml import Component


class Print(Component):
    def transform(self, data: DataFrame) -> DataFrame:
        print(data)
        return data


class Display(Component):
    def transform(self, data: DataFrame) -> DataFrame:
        from IPython.display import display

        display(data)
        return data


class Concat(Component):
    def __init__(
        self,
        components: Iterable[Component],
        *,
        how: ConcatMethod = "vertical",
        rechunk: bool = False,
        parallel: bool = True,
    ):
        self.components = components
        self.how: ConcatMethod = how
        self.rechunk = rechunk
        self.parallel = parallel

    def fit(
        self,
        data: DataFrame,
        validation_data: pl.DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        for component in self.components:
            component.fit(data, validation_data)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        data_list = [component.transform(data) for component in self.components]
        return pl.concat(
            data_list, how=self.how, rechunk=self.rechunk, parallel=self.parallel
        )

    def fit_transform(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> DataFrame:
        data_list = [
            component.fit_transform(data, validation_data)
            for component in self.components
        ]
        return pl.concat(
            data_list, how=self.how, rechunk=self.rechunk, parallel=self.parallel
        )


class SortColumns(Component):
    def __init__(
        self, by: Literal["dtype", "name"] = "dtype", *, descending: bool = False
    ):
        self.by = by
        self.descending = descending

    def transform(self, data: DataFrame) -> DataFrame:
        schema = data.collect_schema()
        sorted_columns = sorted(
            [{"name": k, "dtype": str(v) + k} for k, v in schema.items()],
            key=lambda x: x[self.by],
            reverse=self.descending,
        )
        return data.select([col["name"] for col in sorted_columns])


class GroupByThen(Component):
    def __init__(
        self,
        by: str | Expr | Sequence[str | Expr] | None = None,
        *aggs: IntoExpr | Iterable[IntoExpr],
        maintain_order: bool = False,
    ):
        self.by = by
        self.aggs = aggs
        self.maintain_order = maintain_order

    def fit(
        self,
        data: pl.DataFrame,
        validation_data: pl.DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        self.grouped = data.group_by(self.by, maintain_order=self.maintain_order).agg(
            *self.aggs
        )
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        return data.join(self.grouped, on=self.by, how="left")
