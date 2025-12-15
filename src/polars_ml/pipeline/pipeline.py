from datetime import timedelta
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    Literal,
    Mapping,
    Self,
    Sequence,
    overload,
)

import numpy as np
from polars import DataFrame, Expr, Schema, Series
from polars._typing import (
    AsofJoinStrategy,
    ColumnNameOrSelector,
    ConcatMethod,
    CorrelationMethod,
    FillNullStrategy,
    IntoExpr,
    IntoExprColumn,
    JoinStrategy,
    JoinValidation,
    MaintainOrderJoin,
    PivotAgg,
    PolarsDataType,
    PythonDataType,
    QuantileMethod,
    SchemaDict,
    UniqueKeepStrategy,
    UnstackDirection,
)

from polars_ml.base import Transformer
from polars_ml.gbdt import GBDTNameSpace
from polars_ml.preprocessing import (
    ArithmeticSynthesis,
    BoxCoxTransform,
    Discretize,
    LabelEncode,
    LabelEncodeInverseContext,
    MinMaxScale,
    PowerTransformInverseContext,
    RobustScale,
    ScaleInverseContext,
    StandardScale,
    YeoJohnsonTransform,
)

from .basic import Apply, Concat, Const, Echo, Parrot, Side, ToDummies
from .getattr import GetAttr
from .group_by import GroupByNameSpace


class Pipeline(Transformer):
    def __init__(self, *steps: Transformer):
        self.steps: list[Transformer] = list(*steps)

    def pipe(self, step: Transformer) -> Self:
        self.steps.append(step)
        return self

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        for i, step in enumerate(self.steps):
            if i < len(self.steps) - 1:
                data = step.fit_transform(data, **more_data)
                more_data = {k: step.transform(v) for k, v in more_data.items()}
            else:
                step.fit(data, **more_data)
        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        for i, step in enumerate(self.steps):
            data = step.fit_transform(data, **more_data)
            if i < len(self.steps) - 1:
                more_data = {k: step.transform(v) for k, v in more_data.items()}
        return data

    def transform(self, data: DataFrame) -> DataFrame:
        for step in self.steps:
            data = step.transform(data)
        return data

    @property
    def gbdt(self) -> GBDTNameSpace:
        return GBDTNameSpace(self)

    # --- BEGIN AUTO-GENERATED METHODS IN Pipeline ---
    def approx_n_unique(self) -> Self:
        return self.pipe(GetAttr("approx_n_unique"))

    def bottom_k(
        self,
        k: int,
        by: IntoExpr | Iterable[IntoExpr],
        reverse: bool | Sequence[bool] = False,
    ) -> Self:
        return self.pipe(GetAttr("bottom_k", k, by=by, reverse=reverse))

    def cast(
        self,
        dtypes: Mapping[
            ColumnNameOrSelector | PolarsDataType, PolarsDataType | PythonDataType
        ]
        | PolarsDataType,
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
        percentiles: Sequence[float] | float | None = (0.25, 0.5, 0.75),
        interpolation: QuantileMethod = "nearest",
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
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
    ) -> Self:
        return self.pipe(GetAttr("explode", columns, *more_columns))

    def extend(self, other: DataFrame | Transformer) -> Self:
        return self.pipe(GetAttr("extend", other))

    def fill_nan(self, value: Expr | int | float | None) -> Self:
        return self.pipe(GetAttr("fill_nan", value))

    def fill_null(
        self,
        value: Any | Expr | None = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
        matches_supertype: bool = True,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "fill_null", value, strategy, limit, matches_supertype=matches_supertype
            )
        )

    def filter(
        self,
        *predicates: IntoExprColumn
        | Iterable[IntoExprColumn]
        | bool
        | list[bool]
        | np.ndarray[Any, Any],
        **constraints: Any,
    ) -> Self:
        return self.pipe(GetAttr("filter", *predicates, **constraints))

    def gather_every(self, n: int, offset: int = 0) -> Self:
        return self.pipe(GetAttr("gather_every", n, offset))

    def head(self, n: int = 5) -> Self:
        return self.pipe(GetAttr("head", n))

    def hstack(self, columns: list[Series] | DataFrame, in_place: bool = False) -> Self:
        return self.pipe(GetAttr("hstack", columns, in_place=in_place))

    def insert_column(self, index: int, column: IntoExprColumn) -> Self:
        return self.pipe(GetAttr("insert_column", index, column))

    def interpolate(self) -> Self:
        return self.pipe(GetAttr("interpolate"))

    def join(
        self,
        other: DataFrame | Transformer,
        on: str | Expr | Sequence[str | Expr] | None = None,
        how: JoinStrategy = "inner",
        left_on: str | Expr | Sequence[str | Expr] | None = None,
        right_on: str | Expr | Sequence[str | Expr] | None = None,
        suffix: str = "_right",
        validate: JoinValidation = "m:m",
        nulls_equal: bool = False,
        coalesce: bool | None = None,
        maintain_order: MaintainOrderJoin | None = None,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "join",
                other,
                on,
                how,
                left_on=left_on,
                right_on=right_on,
                suffix=suffix,
                validate=validate,
                nulls_equal=nulls_equal,
                coalesce=coalesce,
                maintain_order=maintain_order,
            )
        )

    def join_asof(
        self,
        other: DataFrame | Transformer,
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
        allow_exact_matches: bool = True,
        check_sortedness: bool = True,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "join_asof",
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
                allow_exact_matches=allow_exact_matches,
                check_sortedness=check_sortedness,
            )
        )

    def join_where(
        self,
        other: DataFrame | Transformer,
        *predicates: Expr | Iterable[Expr],
        suffix: str = "_right",
    ) -> Self:
        return self.pipe(GetAttr("join_where", other, *predicates, suffix=suffix))

    def limit(self, n: int = 5) -> Self:
        return self.pipe(GetAttr("limit", n))

    def map_rows(
        self,
        function: Callable[[tuple[Any, ...]], Any],
        return_dtype: PolarsDataType | None = None,
        inference_size: int = 256,
    ) -> Self:
        return self.pipe(
            GetAttr("map_rows", function, return_dtype, inference_size=inference_size)
        )

    def match_to_schema(
        self,
        schema: SchemaDict | Schema,
        missing_columns: Literal["insert", "raise"]
        | Mapping[str, Literal["insert", "raise"] | Expr] = "raise",
        missing_struct_fields: Literal["insert", "raise"]
        | Mapping[str, Literal["insert", "raise"]] = "raise",
        extra_columns: Literal["ignore", "raise"] = "raise",
        extra_struct_fields: Literal["ignore", "raise"]
        | Mapping[str, Literal["ignore", "raise"]] = "raise",
        integer_cast: Literal["upcast", "forbid"]
        | Mapping[str, Literal["upcast", "forbid"]] = "forbid",
        float_cast: Literal["upcast", "forbid"]
        | Mapping[str, Literal["upcast", "forbid"]] = "forbid",
    ) -> Self:
        return self.pipe(
            GetAttr(
                "match_to_schema",
                schema,
                missing_columns=missing_columns,
                missing_struct_fields=missing_struct_fields,
                extra_columns=extra_columns,
                extra_struct_fields=extra_struct_fields,
                integer_cast=integer_cast,
                float_cast=float_cast,
            )
        )

    def max(self) -> Self:
        return self.pipe(GetAttr("max"))

    def mean(self) -> Self:
        return self.pipe(GetAttr("mean"))

    def median(self) -> Self:
        return self.pipe(GetAttr("median"))

    def melt(
        self,
        id_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        value_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        variable_name: str | None = None,
        value_name: str | None = None,
    ) -> Self:
        return self.pipe(
            GetAttr("melt", id_vars, value_vars, variable_name, value_name)
        )

    def merge_sorted(self, other: DataFrame | Transformer, key: str) -> Self:
        return self.pipe(GetAttr("merge_sorted", other, key))

    def min(self) -> Self:
        return self.pipe(GetAttr("min"))

    def null_count(self) -> Self:
        return self.pipe(GetAttr("null_count"))

    def pivot(
        self,
        on: ColumnNameOrSelector | Sequence[ColumnNameOrSelector],
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
        self, quantile: float, interpolation: QuantileMethod = "nearest"
    ) -> Self:
        return self.pipe(GetAttr("quantile", quantile, interpolation))

    def rechunk(self) -> Self:
        return self.pipe(GetAttr("rechunk"))

    def remove(
        self,
        *predicates: IntoExprColumn
        | Iterable[IntoExprColumn]
        | bool
        | list[bool]
        | np.ndarray[Any, Any],
        **constraints: Any,
    ) -> Self:
        return self.pipe(GetAttr("remove", *predicates, **constraints))

    def rename(
        self, mapping: Mapping[str, str] | Callable[[str], str], strict: bool = True
    ) -> Self:
        return self.pipe(GetAttr("rename", mapping, strict=strict))

    def replace_column(self, index: int, column: Series) -> Self:
        return self.pipe(GetAttr("replace_column", index, column))

    def reverse(self) -> Self:
        return self.pipe(GetAttr("reverse"))

    def sample(
        self,
        n: int | Series | None = None,
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

    def set_sorted(self, column: str, descending: bool = False) -> Self:
        return self.pipe(GetAttr("set_sorted", column, descending=descending))

    def shift(self, n: int = 1, fill_value: IntoExpr | None = None) -> Self:
        return self.pipe(GetAttr("shift", n, fill_value=fill_value))

    def shrink_to_fit(self, in_place: bool = False) -> Self:
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

    def sql(self, query: str, table_name: str = "self") -> Self:
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
        by: IntoExpr | Iterable[IntoExpr],
        reverse: bool | Sequence[bool] = False,
    ) -> Self:
        return self.pipe(GetAttr("top_k", k, by=by, reverse=reverse))

    def transpose(
        self,
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
        separator: str | None = None,
    ) -> Self:
        return self.pipe(GetAttr("unnest", columns, *more_columns, separator=separator))

    def unpivot(
        self,
        on: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
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
        return self.pipe(
            GetAttr(
                "unstack", step=step, how=how, columns=columns, fill_values=fill_values
            )
        )

    def update(
        self,
        other: DataFrame | Transformer,
        on: str | Sequence[str] | None = None,
        how: Literal["left", "inner", "full"] = "left",
        left_on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        include_nulls: bool = False,
        maintain_order: MaintainOrderJoin | None = "left",
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
                maintain_order=maintain_order,
            )
        )

    def upsample(
        self,
        time_column: str,
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

    def vstack(self, other: DataFrame | Transformer, in_place: bool = False) -> Self:
        return self.pipe(GetAttr("vstack", other, in_place=in_place))

    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.pipe(GetAttr("with_columns", *exprs, **named_exprs))

    def with_columns_seq(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.pipe(GetAttr("with_columns_seq", *exprs, **named_exprs))

    def with_row_count(self, name: str = "row_nr", offset: int = 0) -> Self:
        return self.pipe(GetAttr("with_row_count", name, offset))

    def with_row_index(self, name: str = "index", offset: int = 0) -> Self:
        return self.pipe(GetAttr("with_row_index", name, offset))

    def group_by(
        self,
        *by: IntoExpr | Iterable[IntoExpr],
        maintain_order: bool = False,
        **named_by: IntoExpr,
    ) -> GroupByNameSpace:
        return GroupByNameSpace(
            self, "group_by", *by, maintain_order=maintain_order, **named_by
        )

    def apply(self, func: Callable[[DataFrame], DataFrame]) -> Self:
        return self.pipe(Apply(func))

    def const(self, data: DataFrame) -> Self:
        return self.pipe(Const(data))

    def echo(self) -> Self:
        return self.pipe(Echo())

    def parrot(self) -> Self:
        return self.pipe(Parrot())

    def side(self, transformer: Transformer) -> Self:
        return self.pipe(Side(transformer))

    def discretize(
        self,
        exprs: IntoExpr | Iterable[IntoExpr],
        *more_exprs: IntoExpr | Iterable[IntoExpr],
        quantiles: Sequence[float] | int,
        labels: Sequence[str] | None = None,
        left_closed: bool = False,
        allow_duplicates: bool = False,
        suffix: str = "_discretized",
    ) -> Self:
        return self.pipe(
            Discretize(
                exprs,
                *more_exprs,
                quantiles=quantiles,
                labels=labels,
                left_closed=left_closed,
                allow_duplicates=allow_duplicates,
                suffix=suffix,
            )
        )

    def concat(
        self,
        items: Sequence[Transformer],
        how: ConcatMethod = "vertical",
        rechunk: bool = False,
        parallel: bool = True,
    ) -> Self:
        return self.pipe(Concat(items, how=how, rechunk=rechunk, parallel=parallel))

    def to_dummies(
        self,
        columns: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        separator: str = "_",
        drop_first: bool = False,
    ) -> Self:
        return self.pipe(ToDummies(columns, separator=separator, drop_first=drop_first))

    def arithmetic_synthesis(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        order: int,
        method: Literal["additive", "multiplicative"] = "additive",
        drop_high_correlation_features_method: CorrelationMethod | None = None,
        threshold: float = 0.9,
        show_progress: bool = True,
    ) -> Self:
        return self.pipe(
            ArithmeticSynthesis(
                columns,
                order=order,
                method=method,
                drop_high_correlation_features_method=drop_high_correlation_features_method,
                threshold=threshold,
                show_progress=show_progress,
            )
        )

    @overload
    def min_max_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
    ) -> Self: ...

    @overload
    def min_max_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
        inverse_mapping: Mapping[str, str] | None,
    ) -> ScaleInverseContext: ...

    def min_max_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
        inverse_mapping: Mapping[str, str] | None = None,
    ) -> Self | ScaleInverseContext:
        if inverse_mapping is None:
            return self.pipe(MinMaxScale(columns, *more_columns, by=by, suffix=suffix))
        else:
            return ScaleInverseContext(
                self,
                MinMaxScale(columns, *more_columns, by=by, suffix=suffix),
                inverse_mapping,
            )

    @overload
    def standard_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
    ) -> Self: ...

    @overload
    def standard_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
        inverse_mapping: Mapping[str, str] | None,
    ) -> ScaleInverseContext: ...

    def standard_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
        inverse_mapping: Mapping[str, str] | None = None,
    ) -> Self | ScaleInverseContext:
        if inverse_mapping is None:
            return self.pipe(
                StandardScale(columns, *more_columns, by=by, suffix=suffix)
            )
        else:
            return ScaleInverseContext(
                self,
                StandardScale(columns, *more_columns, by=by, suffix=suffix),
                inverse_mapping,
            )

    @overload
    def robust_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        quantile_range: tuple[float, float] = (0.25, 0.75),
        suffix: str = "",
    ) -> Self: ...

    @overload
    def robust_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        quantile_range: tuple[float, float] = (0.25, 0.75),
        suffix: str = "",
        inverse_mapping: Mapping[str, str] | None,
    ) -> ScaleInverseContext: ...

    def robust_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        quantile_range: tuple[float, float] = (0.25, 0.75),
        suffix: str = "",
        inverse_mapping: Mapping[str, str] | None = None,
    ) -> Self | ScaleInverseContext:
        if inverse_mapping is None:
            return self.pipe(
                RobustScale(
                    columns,
                    *more_columns,
                    by=by,
                    quantile_range=quantile_range,
                    suffix=suffix,
                )
            )
        else:
            return ScaleInverseContext(
                self,
                RobustScale(
                    columns,
                    *more_columns,
                    by=by,
                    quantile_range=quantile_range,
                    suffix=suffix,
                ),
                inverse_mapping,
            )

    @overload
    def box_cox_transform(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
    ) -> Self: ...

    @overload
    def box_cox_transform(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
        inverse_mapping: Mapping[str, str] | None,
    ) -> PowerTransformInverseContext: ...

    def box_cox_transform(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
        inverse_mapping: Mapping[str, str] | None = None,
    ) -> Self | PowerTransformInverseContext:
        if inverse_mapping is None:
            return self.pipe(
                BoxCoxTransform(columns, *more_columns, by=by, suffix=suffix)
            )
        else:
            return PowerTransformInverseContext(
                self,
                BoxCoxTransform(columns, *more_columns, by=by, suffix=suffix),
                inverse_mapping,
            )

    @overload
    def yeo_johnson_transform(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
    ) -> Self: ...

    @overload
    def yeo_johnson_transform(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
        inverse_mapping: Mapping[str, str] | None,
    ) -> PowerTransformInverseContext: ...

    def yeo_johnson_transform(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
        inverse_mapping: Mapping[str, str] | None = None,
    ) -> Self | PowerTransformInverseContext:
        if inverse_mapping is None:
            return self.pipe(
                YeoJohnsonTransform(columns, *more_columns, by=by, suffix=suffix)
            )
        else:
            return PowerTransformInverseContext(
                self,
                YeoJohnsonTransform(columns, *more_columns, by=by, suffix=suffix),
                inverse_mapping,
            )

    @overload
    def label_encode(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        orders: Mapping[str, Sequence[Any]] | None = None,
        maintain_order: bool = True,
        suffix: str = "",
    ) -> Self: ...

    @overload
    def label_encode(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        orders: Mapping[str, Sequence[Any]] | None = None,
        maintain_order: bool = True,
        suffix: str = "",
        inverse_mapping: Mapping[str, str] | None,
    ) -> LabelEncodeInverseContext: ...

    def label_encode(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        orders: Mapping[str, Sequence[Any]] | None = None,
        maintain_order: bool = True,
        suffix: str = "",
        inverse_mapping: Mapping[str, str] | None = None,
    ) -> Self | LabelEncodeInverseContext:
        if inverse_mapping is None:
            return self.pipe(
                LabelEncode(
                    columns,
                    *more_columns,
                    orders=orders,
                    maintain_order=maintain_order,
                    suffix=suffix,
                )
            )
        else:
            return LabelEncodeInverseContext(
                self,
                LabelEncode(
                    columns,
                    *more_columns,
                    orders=orders,
                    maintain_order=maintain_order,
                    suffix=suffix,
                ),
                inverse_mapping,
            )

    # --- END AUTO-GENERATED METHODS IN Pipeline ---
