import uuid
from typing import Any, Iterable, Literal, Mapping, Self, Sequence

import polars as pl
from polars import DataFrame, Expr
from polars._typing import IntoExpr

from polars_ml.pipeline.component import PipelineComponent


class Echo(PipelineComponent):
    def transform(self, data: DataFrame) -> DataFrame:
        return data


class GetAttr(PipelineComponent):
    def __init__(self, method: str, *args: Any, **kwargs: Any):
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def transform(self, data: DataFrame) -> DataFrame:
        return getattr(data, self.method)(*self.args, **self.kwargs)


class Print(PipelineComponent):
    def transform(self, data: DataFrame) -> DataFrame:
        print(data)
        return data


class Display(PipelineComponent):
    def transform(self, data: DataFrame) -> DataFrame:
        from IPython.display import display

        display(data)
        return data


class SortColumns(PipelineComponent):
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


class GroupByThen(PipelineComponent):
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
    ) -> Self:
        self.grouped = data.group_by(self.by, maintain_order=self.maintain_order).agg(
            *self.aggs
        )
        if self.after_with_columns is not None:
            self.grouped = self.grouped.with_columns(self.after_with_columns)

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        return data.join(self.grouped, on=self.by, how="left")


class Impute(PipelineComponent):
    def __init__(
        self,
        imputer: PipelineComponent,
        column: str,
        *,
        maintain_order: bool = False,
    ):
        self.imputer = imputer
        self.column = column
        self.maintain_order = maintain_order

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
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
            imputed_data = self.imputer.transform(
                missing_data.drop(self.column, index_name)
            )
            filled_data = missing_data.with_columns(imputed_data[self.column])
            data = pl.concat(
                [data.filter(pl.col(self.column).is_not_null()), filled_data]
            )
            return data.sort(index_name).drop(index_name)
        else:
            missing_data = data.filter(pl.col(self.column).is_null())
            imputed_data = self.imputer.transform(missing_data.drop(self.column))
            filled_data = missing_data.with_columns(imputed_data[self.column])
            data = pl.concat(
                [data.filter(pl.col(self.column).is_not_null()), filled_data]
            )
            return data
