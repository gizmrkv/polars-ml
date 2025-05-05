from datetime import timedelta
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping, Sequence

import polars as pl
from polars import DataFrame, Expr
from polars._typing import (
    AsofJoinStrategy,
    ColumnNameOrSelector,
    ConcatMethod,
    IntoExpr,
    JoinStrategy,
    JoinValidation,
    MaintainOrderJoin,
)

from .component import Component


class Extend(Component):
    def __init__(self, other: DataFrame | Component):
        self.other = other

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> "Extend":
        if isinstance(self.other, Component):
            self.other.fit(data, validation_data)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.transform(data)
        else:
            other = self.other

        return data.extend(other)

    def fit_transform(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.fit_transform(data, validation_data)
        else:
            other = self.other

        return data.extend(other)


class Join(Component):
    def __init__(
        self,
        other: DataFrame,
        on: str | Expr | Sequence[str | Expr] | None = None,
        how: JoinStrategy = "inner",
        *,
        left_on: str | Expr | Sequence[str | Expr] | None = None,
        right_on: str | Expr | Sequence[str | Expr] | None = None,
        suffix: str = "_right",
        validate: JoinValidation = "m:m",
        nulls_equal: bool = False,
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
        self.nulls_equal = nulls_equal
        self.join_nulls = False
        self.coalesce = coalesce
        self.maintain_order: MaintainOrderJoin | None = maintain_order

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> "Join":
        if isinstance(self.other, Component):
            self.other.fit(data, validation_data)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.transform(data)
        else:
            other = self.other

        return self._join(data, other)

    def fit_transform(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.fit_transform(data, validation_data)
        else:
            other = self.other

        return self._join(data, other)

    def _join(self, data: DataFrame, other: DataFrame) -> DataFrame:
        return data.join(
            other,
            on=self.on,
            how=self.how,
            left_on=self.left_on,
            right_on=self.right_on,
            suffix=self.suffix,
            validate=self.validate,
            nulls_equal=self.nulls_equal,
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
    ) -> "JoinAsof":
        if isinstance(self.other, Component):
            self.other.fit(data, validation_data)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.transform(data)
        else:
            other = self.other

        return self._join(data, other)

    def fit_transform(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.fit_transform(data, validation_data)
        else:
            other = self.other

        return self._join(data, other)

    def _join(self, data: DataFrame, other: DataFrame) -> DataFrame:
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
    ) -> "JoinWhere":
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
    ) -> "MergeSorted":
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

        return data.merge_sorted(other, self.key)


class Update(Component):
    def __init__(
        self,
        other: DataFrame | Component,
        on: str | Sequence[str] | None = None,
        how: Literal["left", "inner", "full"] = "left",
        *,
        left_on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        include_nulls: bool = False,
    ):
        self.other = other
        self.on = on
        self.how: Literal["left", "inner", "full"] = how
        self.left_on = left_on
        self.right_on = right_on
        self.include_nulls = include_nulls

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> "Update":
        if isinstance(self.other, Component):
            self.other.fit(data, validation_data)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.transform(data)
        else:
            other = self.other

        return self._update(data, other)

    def fit_transform(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> DataFrame:
        if isinstance(self.other, Component):
            other = self.other.fit_transform(data, validation_data)
        else:
            other = self.other

        return self._update(data, other)

    def _update(self, data: DataFrame, other: DataFrame) -> DataFrame:
        return data.update(
            other,
            self.on,
            self.how,
            left_on=self.left_on,
            right_on=self.right_on,
            include_nulls=self.include_nulls,
        )


class VStack(Component):
    def __init__(self, other: DataFrame | Component, *, in_place: bool = False):
        self.other = other
        self.in_place = in_place

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> "VStack":
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


class Concat(Component):
    def __init__(
        self,
        *others: DataFrame | Component,
        how: ConcatMethod = "vertical",
        rechunk: bool = False,
        parallel: bool = True,
        include_input: bool = False,
    ):
        self.others = others
        self.how: ConcatMethod = how
        self.rechunk = rechunk
        self.parallel = parallel
        self.include_input = include_input

    def fit(
        self,
        data: DataFrame,
        validation_data: pl.DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> "Concat":
        for other in self.others:
            if isinstance(other, Component):
                other.fit(data, validation_data)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        data_list = [
            other.transform(data) if isinstance(other, Component) else other
            for other in self.others
        ]
        if self.include_input:
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
            other.fit_transform(data, validation_data)
            if isinstance(other, Component)
            else other
            for other in self.others
        ]
        if self.include_input:
            data_list = [data] + data_list

        return pl.concat(
            data_list, how=self.how, rechunk=self.rechunk, parallel=self.parallel
        )
