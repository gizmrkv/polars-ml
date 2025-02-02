from datetime import timedelta
from typing import Any, Iterable, Literal, Mapping, Self, Sequence

import polars as pl
from polars import DataFrame, Expr
from polars._typing import (
    AsofJoinStrategy,
    ConcatMethod,
    JoinStrategy,
    JoinValidation,
    MaintainOrderJoin,
)

from polars_ml import Component


class GetAttr(Component):
    def __init__(self, method: str, *args: Any, **kwargs: Any):
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def transform(self, data: DataFrame) -> DataFrame:
        return getattr(data, self.method)(*self.args, **self.kwargs)


class Join(Component):
    def __init__(
        self,
        other: DataFrame | Component,
        on: str | Expr | Sequence[str | Expr] | None = None,
        how: JoinStrategy = "inner",
        *,
        left_on: str | Expr | Sequence[str | Expr] | None = None,
        right_on: str | Expr | Sequence[str | Expr] | None = None,
        suffix: str = "_right",
        validate: JoinValidation = "m:m",
        join_nulls: bool = False,
        coalesce: bool | None = None,
        maintain_order: MaintainOrderJoin | None = None,
    ):
        self.other = other
        self.on = on
        self.how: JoinStrategy = how
        self.left_on = left_on
        self.right_on = right_on
        self.suffix = suffix
        self.validate: JoinValidation = validate
        self.join_nulls = join_nulls
        self.coalesce = coalesce
        self.maintain_order: MaintainOrderJoin | None = maintain_order

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        if isinstance(self.other, Component):
            self.other.fit(data, validation_data)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.transform(data)
        else:
            other = self.other

        return data.join(
            other,
            self.on,
            self.how,
            left_on=self.left_on,
            right_on=self.right_on,
            suffix=self.suffix,
            validate=self.validate,
            join_nulls=self.join_nulls,
            coalesce=self.coalesce,
            maintain_order=self.maintain_order,
        )

    def fit_transform(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.fit_transform(data, validation_data)
        else:
            other = self.other

        return data.join(
            other,
            self.on,
            self.how,
            left_on=self.left_on,
            right_on=self.right_on,
            suffix=self.suffix,
            validate=self.validate,
            join_nulls=self.join_nulls,
            coalesce=self.coalesce,
            maintain_order=self.maintain_order,
        )


class JoinAsof(Component):
    def __init__(
        self,
        other: DataFrame | Component,
        *,
        left_on: str | None | Expr = None,
        right_on: str | None | Expr = None,
        on: str | None | Expr = None,
        by_left: str | Sequence[str] | None = None,
        by_right: str | Sequence[str] | None = None,
        by: str | Sequence[str] | None = None,
        strategy: AsofJoinStrategy = "backward",
        suffix: str = "_right",
        tolerance: str | int | float | timedelta | None = None,
        allow_parallel: bool = True,
        force_parallel: bool = False,
        coalesce: bool = True,
    ):
        self.other = other
        self.left_on = left_on
        self.right_on = right_on
        self.on = on
        self.by_left = by_left
        self.by_right = by_right
        self.by = by
        self.strategy: AsofJoinStrategy = strategy
        self.suffix = suffix
        self.tolerance = tolerance
        self.allow_parallel = allow_parallel
        self.force_parallel = force_parallel
        self.coalesce = coalesce

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        if isinstance(self.other, Component):
            self.other.fit(data, validation_data)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.transform(data)
        else:
            other = self.other

        return data.join_asof(
            other,
            left_on=self.left_on,
            right_on=self.right_on,
            on=self.on,
            by_left=self.by_left,
            by_right=self.by_right,
            by=self.by,
            strategy=self.strategy,
            suffix=self.suffix,
            tolerance=self.tolerance,
            allow_parallel=self.allow_parallel,
            force_parallel=self.force_parallel,
            coalesce=self.coalesce,
        )

    def fit_transform(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.fit_transform(data, validation_data)
        else:
            other = self.other

        return data.join_asof(
            other,
            left_on=self.left_on,
            right_on=self.right_on,
            on=self.on,
            by_left=self.by_left,
            by_right=self.by_right,
            by=self.by,
            strategy=self.strategy,
            suffix=self.suffix,
            tolerance=self.tolerance,
            allow_parallel=self.allow_parallel,
            force_parallel=self.force_parallel,
            coalesce=self.coalesce,
        )


class JoinWhere(Component):
    def __init__(
        self,
        other: DataFrame | Component,
        *predicates: Expr | Iterable[Expr],
        suffix: str = "_right",
    ):
        self.other = other
        self.predicates = predicates
        self.suffix = suffix

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        if isinstance(self.other, Component):
            self.other.fit(data, validation_data)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.transform(data)
        else:
            other = self.other

        return data.join_where(other, *self.predicates, suffix=self.suffix)

    def fit_transform(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.fit_transform(data, validation_data)
        else:
            other = self.other

        return data.join_where(other, *self.predicates, suffix=self.suffix)


class MergeSorted(Component):
    def __init__(self, other: DataFrame | Component, key: str):
        self.other = other
        self.key = key

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        if isinstance(self.other, Component):
            self.other.fit(data, validation_data)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.transform(data)
        else:
            other = self.other

        return data.merge_sorted(other, self.key)

    def fit_transform(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.fit_transform(data, validation_data)
        else:
            other = self.other

        data.vstack
        return data.merge_sorted(other, self.key)


class VStack(Component):
    def __init__(self, other: DataFrame | Component, *, in_place: bool = False):
        self.other = other
        self.in_place = in_place

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        if isinstance(self.other, Component):
            self.other.fit(data, validation_data)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.transform(data)
        else:
            other = self.other
        return data.vstack(other, in_place=self.in_place)

    def fit_transform(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.fit_transform(data, validation_data)
        else:
            other = self.other
        return data.vstack(other, in_place=self.in_place)


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
        *components: Component,
        how: ConcatMethod = "vertical",
        rechunk: bool = False,
        parallel: bool = True,
        append_output: bool = True,
    ):
        self.components = components
        self.how: ConcatMethod = how
        self.rechunk = rechunk
        self.parallel = parallel
        self.append_output = append_output

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
        if self.append_output:
            data_list = [data] + data_list

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
        if self.append_output:
            data_list = [data] + data_list

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
