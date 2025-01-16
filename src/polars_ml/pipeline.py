from datetime import timedelta
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Iterable,
    Literal,
    Mapping,
    Self,
    Sequence,
)

import numpy as np
from polars import DataFrame, Expr, Series
from polars._typing import (
    AsofJoinStrategy,
    ClosedInterval,
    ColumnNameOrSelector,
    ConcatMethod,
    FillNullStrategy,
    IntoExpr,
    IntoExprColumn,
    JoinStrategy,
    JoinValidation,
    Label,
    MaintainOrderJoin,
    PivotAgg,
    PolarsDataType,
    PythonDataType,
    RollingInterpolationMethod,
    StartBy,
    UniqueKeepStrategy,
    UnstackDirection,
)

from .component import Component
from .group_by import DynamicGroupBy, GroupByNamaSpace, RollingGroupBy
from .horizontal import (
    HorizontalAgg,
    HorizontalAll,
    HorizontalArgMax,
    HorizontalArgMin,
    HorizontalCount,
    HorizontalMax,
    HorizontalMean,
    HorizontalMedian,
    HorizontalMin,
    HorizontalNUnique,
    HorizontalQuantile,
    HorizontalSum,
)
from .transformer import (
    Binning,
    LabelEncoding,
    MinMaxScaler,
    QuantileScaler,
    StandardScaler,
)
from .utils import (
    Concat,
    Display,
    GetAttr,
    GroupByThen,
    Impute,
    Join,
    JoinAsof,
    JoinWhere,
    MergeSorted,
    Print,
    SortColumns,
)

if TYPE_CHECKING:
    from .model import DecompositionNameSpace, LinearNameSpace, TreeNameSpace


class Pipeline(Component):
    def __init__(self):
        self.components: list[Component] = []

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        for component in self.components[:-1]:
            data = component.fit_transform(data, validation_data)
            if validation_data is None:
                continue

            if isinstance(validation_data, DataFrame):
                validation_data = component.transform(validation_data)
            else:
                validation_data = {
                    name: component.transform(validation_data)
                    for name, validation_data in validation_data.items()
                }

        self.components[-1].fit(data)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        for component in self.components:
            data = component.transform(data)
        return data

    def fit_transform(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> DataFrame:
        for component in self.components:
            data = component.fit_transform(data, validation_data)
            if validation_data is None:
                continue

            if isinstance(validation_data, DataFrame):
                validation_data = component.transform(validation_data)
            else:
                validation_data = {
                    name: component.transform(validation_data)
                    for name, validation_data in validation_data.items()
                }

        return data

    def pipe(self, component: Component) -> Self:
        self.components.append(component)
        return self

    def approx_n_unique(self) -> Self:
        return self.pipe(GetAttr("approx_n_unique"))

    def bottom_k(
        self,
        k: int,
        *,
        by: IntoExpr | Iterable[IntoExpr],
        reverse: bool | Sequence[bool] = False,
    ) -> Self:
        return self.pipe(GetAttr("bottom_k", k, by=by, reverse=reverse))

    def cast(
        self,
        dtypes: (
            Mapping[
                ColumnNameOrSelector | PolarsDataType, PolarsDataType | PythonDataType
            ]
            | PolarsDataType
        ),
        *,
        strict: bool = True,
    ) -> Self:
        return self.pipe(GetAttr("cast", dtypes, strict=strict))

    def clear(self, n: int = 0) -> Self:
        return self.pipe(GetAttr("clear", n))

    def clone(self) -> Self:
        return self.pipe(GetAttr("clone"))

    def corr(self, **kwargs: Any) -> Self:
        return self.pipe(GetAttr("corr", **kwargs))

    def count(self) -> Self:
        return self.pipe(GetAttr("count"))

    def describe(
        self,
        percentiles: Sequence[float] | float | None = (0.25, 0.50, 0.75),
        *,
        interpolation: RollingInterpolationMethod = "nearest",
    ) -> Self:
        return self.pipe(GetAttr("describe", percentiles, interpolation=interpolation))

    def drop(
        self,
        *columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        strict: bool = True,
    ) -> Self:
        return self.pipe(GetAttr("drop", *columns, strict=strict))

    def drop_nans(
        self,
        subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
    ) -> Self:
        return self.pipe(GetAttr("drop_nans", subset))

    def drop_nulls(
        self,
        subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
    ) -> Self:
        return self.pipe(GetAttr("drop_nulls", subset))

    def explode(
        self,
        columns: str | Expr | Sequence[str | Expr],
        *more_columns: str | Expr,
    ) -> Self:
        return self.pipe(GetAttr("explode", columns, *more_columns))

    def extend(self, other: DataFrame) -> Self:
        return self.pipe(GetAttr("extend", other))

    def fill_nan(self, value: Expr | int | float | None) -> Self:
        return self.pipe(GetAttr("fill_nan", value))

    def fill_null(
        self,
        value: Any | Expr | None = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
        *,
        matches_supertype: bool = True,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "fill_null", value, strategy, limit, matches_supertype=matches_supertype
            )
        )

    def filter(
        self,
        *predicates: (
            IntoExprColumn
            | Iterable[IntoExprColumn]
            | bool
            | list[bool]
            | np.ndarray[Any, Any]
        ),
        **constraints: Any,
    ) -> Self:
        return self.pipe(GetAttr("filter", *predicates, **constraints))

    def gather_every(self, n: int, offset: int = 0) -> Self:
        return self.pipe(GetAttr("gather_every", n, offset))

    def group_by(
        self,
        *by: IntoExpr | Iterable[IntoExpr],
        maintain_order: bool = False,
        **named_by: IntoExpr,
    ) -> GroupByNamaSpace:
        return GroupByNamaSpace(
            self, "group_by", *by, maintain_order=maintain_order, **named_by
        )

    def group_by_dynamic(
        self,
        index_column: IntoExpr,
        *,
        every: str | timedelta,
        period: str | timedelta | None = None,
        offset: str | timedelta | None = None,
        include_boundaries: bool = False,
        closed: ClosedInterval = "left",
        label: Label = "left",
        group_by: IntoExpr | Iterable[IntoExpr] | None = None,
        start_by: StartBy = "window",
    ) -> DynamicGroupBy:
        return DynamicGroupBy(
            self,
            "group_by_dynamic",
            index_column,
            every=every,
            period=period,
            offset=offset,
            include_boundaries=include_boundaries,
            closed=closed,
            label=label,
            group_by=group_by,
            start_by=start_by,
        )

    def head(self, n: int = 5) -> Self:
        return self.pipe(GetAttr("head", n))

    def insert_column(self, index: int, column: IntoExprColumn) -> Self:
        return self.pipe(GetAttr("insert_column", index, column))

    def interpolate(self) -> Self:
        return self.pipe(GetAttr("interpolate"))

    def join(
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
    ) -> Self:
        return self.pipe(
            Join(
                other,
                on,
                how,
                left_on=left_on,
                right_on=right_on,
                suffix=suffix,
                validate=validate,
                join_nulls=join_nulls,
                coalesce=coalesce,
                maintain_order=maintain_order,
            )
        )

    def join_asof(
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
    ) -> Self:
        return self.pipe(
            JoinAsof(
                other,
                left_on=left_on,
                right_on=right_on,
                on=on,
                by_left=by_left,
                by_right=by_right,
                by=by,
                strategy=strategy,
                suffix=suffix,
                tolerance=tolerance,
                allow_parallel=allow_parallel,
                force_parallel=force_parallel,
                coalesce=coalesce,
            )
        )

    def join_where(
        self,
        other: DataFrame | Component,
        *predicates: Expr | Iterable[Expr],
        suffix: str = "_right",
    ) -> Self:
        return self.pipe(JoinWhere(other, *predicates, suffix=suffix))

    def limit(self, n: int = 5) -> Self:
        return self.pipe(GetAttr("limit", n))

    def map_rows(
        self,
        function: Callable[[tuple[Any, ...]], Any],
        return_dtype: PolarsDataType | None = None,
        *,
        inference_size: int = 256,
    ) -> Self:
        return self.pipe(GetAttr("map_rows", function, return_dtype, inference_size))

    def mean(self) -> Self:
        return self.pipe(GetAttr("mean"))

    def median(self) -> Self:
        return self.pipe(GetAttr("median"))

    def merge_sorted(self, other: DataFrame | Component, key: str) -> Self:
        return self.pipe(MergeSorted(other, key))

    def min(self) -> Self:
        return self.pipe(GetAttr("min"))

    def null_count(self) -> Self:
        return self.pipe(GetAttr("null_count"))

    def pivot(
        self,
        on: ColumnNameOrSelector | Sequence[ColumnNameOrSelector],
        *,
        index: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        values: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        aggregate_function: PivotAgg | Expr | None = None,
        maintain_order: bool = True,
        sort_columns: bool = False,
        separator: str = "_",
    ) -> Self:
        return self.pipe(
            GetAttr(
                "pivot",
                on,
                index=index,
                values=values,
                aggregate_function=aggregate_function,
                maintain_order=maintain_order,
                sort_columns=sort_columns,
                separator=separator,
            )
        )

    def product(self) -> Self:
        return self.pipe(GetAttr("product"))

    def quantile(
        self, quantile: float, interpolation: RollingInterpolationMethod = "nearest"
    ) -> Self:
        return self.pipe(GetAttr("quantile", quantile, interpolation))

    def rechunk(self) -> Self:
        return self.pipe(GetAttr("rechunk"))

    def rename(
        self, mapping: dict[str, str] | Callable[[str], str], *, strict: bool = True
    ) -> Self:
        return self.pipe(GetAttr("rename", mapping, strict=strict))

    def replace_column(self, index: int, column: Series) -> Self:
        return self.pipe(GetAttr("replace_column", index, column))

    def reverse(self) -> Self:
        return self.pipe(GetAttr("reverse"))

    def rolling(
        self,
        index_column: IntoExpr,
        *,
        period: str | timedelta,
        offset: str | timedelta | None = None,
        closed: ClosedInterval = "right",
        group_by: IntoExpr | Iterable[IntoExpr] | None = None,
    ) -> RollingGroupBy:
        return RollingGroupBy(
            self,
            "rolling",
            index_column,
            period=period,
            offset=offset,
            closed=closed,
            group_by=group_by,
        )

    def sample(
        self,
        n: int | Series | None = None,
        *,
        fraction: float | Series | None = None,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "sample",
                n,
                fraction=fraction,
                with_replacement=with_replacement,
                shuffle=shuffle,
                seed=seed,
            )
        )

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.pipe(GetAttr("select", *exprs, **named_exprs))

    def select_seq(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.pipe(GetAttr("select_seq", *exprs, **named_exprs))

    def set_sorted(
        self,
        column: str,
        *,
        descending: bool = False,
    ) -> Self:
        return self.pipe(GetAttr("set_sorted", column, descending=descending))

    def shift(self, n: int = 1, *, fill_value: IntoExpr | None = None) -> Self:
        return self.pipe(GetAttr("shift", n, fill_value=fill_value))

    def shrink_to_fit(self, *, in_place: bool = False) -> Self:
        return self.pipe(GetAttr("shrink_to_fit", in_place=in_place))

    def slice(self, offset: int, length: int | None = None) -> Self:
        return self.pipe(GetAttr("slice", offset, length))

    def sort(
        self,
        by: IntoExpr | Iterable[IntoExpr],
        *more_by: IntoExpr,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool | Sequence[bool] = False,
        multithreaded: bool = True,
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "sort",
                by,
                *more_by,
                descending=descending,
                nulls_last=nulls_last,
                multithreaded=multithreaded,
                maintain_order=maintain_order,
            )
        )

    def sql(self, query: str, *, table_name: str = "self") -> Self:
        return self.pipe(GetAttr("sql", query, table_name=table_name))

    def std(self, ddof: int = 1) -> Self:
        return self.pipe(GetAttr("std", ddof))

    def sum(self) -> Self:
        return self.pipe(GetAttr("sum"))

    def tail(self, n: int = 5) -> Self:
        return self.pipe(GetAttr("tail", n))

    def top_k(
        self,
        k: int,
        *,
        by: IntoExpr | Iterable[IntoExpr],
        reverse: bool | Sequence[bool] = False,
    ) -> Self:
        return self.pipe(GetAttr("top_k", k, by=by, reverse=reverse))

    def transpose(
        self,
        *,
        include_header: bool = False,
        header_name: str = "column",
        column_names: str | Iterable[str] | None = None,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "transpose",
                include_header=include_header,
                header_name=header_name,
                column_names=column_names,
            )
        )

    def unique(
        self,
        subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
        *,
        keep: UniqueKeepStrategy = "any",
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            GetAttr("unique", subset, keep=keep, maintain_order=maintain_order)
        )

    def unnest(
        self,
        columns: ColumnNameOrSelector | Collection[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
    ) -> Self:
        return self.pipe(GetAttr("unnest", columns, *more_columns))

    def unpivot(
        self,
        on: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        *,
        index: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        variable_name: str | None = None,
        value_name: str | None = None,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "unpivot",
                on,
                index=index,
                variable_name=variable_name,
                value_name=value_name,
            )
        )

    def unstack(
        self,
        step: int,
        how: UnstackDirection = "vertical",
        columns: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        fill_values: list[Any] | None = None,
    ) -> Self:
        return self.pipe(GetAttr("unstack", step, how, columns, fill_values))

    def update(
        self,
        other: DataFrame,
        on: str | Sequence[str] | None = None,
        how: Literal["left", "inner", "full"] = "left",
        *,
        left_on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        include_nulls: bool = False,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "update",
                other,
                on,
                how,
                left_on=left_on,
                right_on=right_on,
                include_nulls=include_nulls,
            )
        )

    def upsample(
        self,
        time_column: str,
        *,
        every: str | timedelta,
        group_by: str | Sequence[str] | None = None,
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "upsample",
                time_column,
                every=every,
                group_by=group_by,
                maintain_order=maintain_order,
            )
        )

    def var(self, ddof: int = 1) -> Self:
        return self.pipe(GetAttr("var", ddof))

    def vstack(self, other: DataFrame, *, in_place: bool = False) -> Self:
        return self.pipe(GetAttr("vstack", other, in_place=in_place))

    def with_columns(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> Self:
        return self.pipe(GetAttr("with_columns", *exprs, **named_exprs))

    def with_columns_seq(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> Self:
        return self.pipe(GetAttr("with_columns_seq", *exprs, **named_exprs))

    def with_row_index(self, name: str = "index", offset: int = 0) -> Self:
        return self.pipe(GetAttr("with_row_index", name, offset))

    def concat(
        self,
        components: Iterable[Component],
        *,
        how: ConcatMethod = "vertical",
        rechunk: bool = False,
        parallel: bool = True,
    ) -> Self:
        return self.pipe(
            Concat(components, how=how, rechunk=rechunk, parallel=parallel)
        )

    def print(self) -> Self:
        return self.pipe(Print())

    def display(self) -> Self:
        return self.pipe(Display())

    def sort_columns(
        self, by: Literal["dtype", "name"] = "dtype", *, descending: bool = False
    ) -> Self:
        return self.pipe(SortColumns(by, descending=descending))

    def group_by_then(
        self,
        by: str | Expr | Sequence[str | Expr] | None = None,
        *aggs: IntoExpr | Iterable[IntoExpr],
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(GroupByThen(by, *aggs, maintain_order=maintain_order))

    def impute(
        self,
        imputer: Component,
        column: str,
        *,
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(Impute(imputer, column, maintain_order=maintain_order))

    def min_max_scale(self, *expr: IntoExpr | Iterable[IntoExpr]) -> Self:
        return self.pipe(MinMaxScaler(*expr))

    def standard_scale(self, *expr: IntoExpr | Iterable[IntoExpr]) -> Self:
        return self.pipe(StandardScaler(*expr))

    def quantile_scale(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        quantile: tuple[float, float] = (0.25, 0.75),
    ) -> Self:
        return self.pipe(QuantileScaler(*expr, quantile=quantile))

    def label_encode(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        orders: dict[str, Sequence[Any]] | None = None,
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            LabelEncoding(*exprs, orders=orders, maintain_order=maintain_order)
        )

    def binning(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        quantiles: Sequence[float] | int,
        labels: Sequence[str] | None = None,
        left_closed: bool = False,
        allow_duplicates: bool = False,
        suffix: str = "_bin",
    ) -> Self:
        return self.pipe(
            Binning(
                *exprs,
                quantiles=quantiles,
                labels=labels,
                left_closed=left_closed,
                allow_duplicates=allow_duplicates,
                suffix=suffix,
            )
        )

    def horizontal_agg(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_agg",
        maintain_order: bool = False,
        aggs: list[Expr] | None = None,
        named_aggs: dict[str, Expr] | None = None,
    ) -> Self:
        return self.pipe(
            HorizontalAgg(
                *expr,
                value_name=value_name,
                maintain_order=maintain_order,
                aggs=aggs,
                named_aggs=named_aggs,
            )
        )

    def horizontal_all(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_all",
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            HorizontalAll(*expr, value_name=value_name, maintain_order=maintain_order)
        )

    def horizontal_count(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_count",
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            HorizontalCount(*expr, value_name=value_name, maintain_order=maintain_order)
        )

    def horizontal_max(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_max",
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            HorizontalMax(*expr, value_name=value_name, maintain_order=maintain_order)
        )

    def horizontal_mean(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_mean",
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            HorizontalMean(*expr, value_name=value_name, maintain_order=maintain_order)
        )

    def horizontal_median(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_median",
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            HorizontalMedian(
                *expr, value_name=value_name, maintain_order=maintain_order
            )
        )

    def horizontal_min(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_min",
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            HorizontalMin(*expr, value_name=value_name, maintain_order=maintain_order)
        )

    def horizontal_n_unique(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_n_unique",
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            HorizontalNUnique(
                *expr, value_name=value_name, maintain_order=maintain_order
            )
        )

    def horizontal_quantile(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        quantile: float,
        value_name: str = "horizontal_quantile",
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            HorizontalQuantile(
                *expr,
                quantile=quantile,
                value_name=value_name,
                maintain_order=maintain_order,
            )
        )

    def horizontal_sum(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_sum",
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            HorizontalSum(*expr, value_name=value_name, maintain_order=maintain_order)
        )

    def horizontal_argmax(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_argmax",
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            HorizontalArgMax(
                *expr, value_name=value_name, maintain_order=maintain_order
            )
        )

    def horizontal_argmin(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_argmin",
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            HorizontalArgMin(
                *expr, value_name=value_name, maintain_order=maintain_order
            )
        )

    @property
    def tree(self) -> "TreeNameSpace":
        return TreeNameSpace(self)

    @property
    def linear(self) -> "LinearNameSpace":
        return LinearNameSpace(self)

    @property
    def decomposition(self) -> "DecompositionNameSpace":
        return DecompositionNameSpace(self)
