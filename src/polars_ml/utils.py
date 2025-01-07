from typing import Iterable, Literal, Self

import polars as pl
from polars import DataFrame
from polars._typing import ConcatMethod

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

    def fit(self, data: DataFrame) -> Self:
        for component in self.components:
            component.fit(data)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        data_list = [component.transform(data) for component in self.components]
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
