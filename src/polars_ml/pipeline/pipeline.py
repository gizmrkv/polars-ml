from __future__ import annotations

from datetime import datetime, timedelta
from io import IOBase
from pathlib import Path
from typing import (
    IO,
    TYPE_CHECKING,
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
import polars as pl
from polars import DataFrame, Expr, Schema, Series

from polars_ml.base import HasFeatureImportance, Transformer
from polars_ml.feature_engineering import FeatureEngineeringNameSpace
from polars_ml.gbdt import GBDTNameSpace
from polars_ml.linear import LinearNameSpace
from polars_ml.metrics import MetricsNameSpace
from polars_ml.optimize import OptimizeNameSpace

from .basic import Apply, Const, Echo, Replay, Side
from .combine import Combine
from .concat import Concat
from .discretize import Discretize
from .getattr import GetAttr
from .group_by import DynamicGroupByNameSpace, GroupByNameSpace, RollingGroupByNameSpace
from .horizontal import HorizontalNameSpace
from .join_agg import JoinAgg
from .label_encode import LabelEncode, LabelEncodeInverseContext
from .power import BoxCoxTransform, PowerTransformInverseContext, YeoJohnsonTransform
from .scale import MinMaxScale, RobustScale, ScaleInverseContext, StandardScale

if TYPE_CHECKING:
    import deltalake
    from deltalake import DeltaTable
    from polars import Expr, Series
    from polars._typing import (
        AsofJoinStrategy,
        AvroCompression,
        ClosedInterval,
        ColumnFormatDict,
        ColumnNameOrSelector,
        ColumnTotalsDefinition,
        ColumnWidthsDefinition,
        ConcatMethod,
        ConditionalFormatDict,
        ConnectionOrCursor,
        CsvEncoding,
        CsvQuoteStyle,
        DbReadEngine,
        DbWriteEngine,
        DbWriteMode,
        FileSource,
        FillNullStrategy,
        IntoExpr,
        IntoExprColumn,
        IpcCompression,
        JoinStrategy,
        JoinValidation,
        Label,
        MaintainOrderJoin,
        OneOrMoreDataTypes,
        ParallelStrategy,
        ParquetCompression,
        ParquetMetadata,
        PivotAgg,
        PolarsDataType,
        PythonDataType,
        QuantileMethod,
        RowTotalsDefinition,
        SchemaDefinition,
        SchemaDict,
        SelectorType,
        SerializationFormat,
        StartBy,
        UniqueKeepStrategy,
        UnstackDirection,
    )
    from polars.interchange.protocol import CompatLevel
    from polars.io.cloud import CredentialProviderFunction
    from xlsxwriter import Workbook
    from xlsxwriter.worksheet import Worksheet


class Pipeline(Transformer, HasFeatureImportance):
    def __init__(
        self,
        *steps: Transformer,
    ):
        self._steps = list(steps)

    def pipe(self, step: Transformer) -> Self:
        self._steps.append(step)
        return self

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        for i, step in enumerate(self._steps):
            if i < len(self._steps) - 1:
                data = step.fit_transform(data, **more_data)
                more_data = {k: step.transform(v) for k, v in more_data.items()}
            else:
                step.fit(data, **more_data)

        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        for i, step in enumerate(self._steps):
            data = step.fit_transform(data, **more_data)
            if i < len(self._steps) - 1:
                more_data = {k: step.transform(v) for k, v in more_data.items()}

        return data

    def transform(self, data: DataFrame) -> DataFrame:
        for step in self._steps:
            data = step.transform(data)
        return data

    def get_feature_importance(self) -> DataFrame:
        if not self._steps:
            raise ValueError("Pipeline has no steps.")

        last_step = self._steps[-1]
        if isinstance(last_step, HasFeatureImportance):
            return last_step.get_feature_importance()

        raise TypeError(
            f"The last step of the pipeline ({type(last_step).__name__}) "
            "does not support feature importance."
        )

    def __len__(self) -> int:
        return len(self._steps)

    @property
    def gbdt(self) -> GBDTNameSpace:
        return GBDTNameSpace(self)

    @property
    def linear(self) -> LinearNameSpace:
        return LinearNameSpace(self)

    @property
    def metrics(self) -> MetricsNameSpace:
        return MetricsNameSpace(self)

    @property
    def optimize(self) -> OptimizeNameSpace:
        return OptimizeNameSpace(self)

    @property
    def horizontal(self) -> HorizontalNameSpace:
        return HorizontalNameSpace(self)

    @property
    def fe(self) -> FeatureEngineeringNameSpace:
        return FeatureEngineeringNameSpace(self)

    def group_by(
        self,
        *by: IntoExpr | Iterable[IntoExpr],
        maintain_order: bool = False,
        **named_by: IntoExpr,
    ) -> GroupByNameSpace:
        return GroupByNameSpace(self, *by, maintain_order=maintain_order, **named_by)

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
    ) -> DynamicGroupByNameSpace:
        return DynamicGroupByNameSpace(
            self,
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

    def rolling(
        self,
        index_column: IntoExpr,
        *,
        period: str | timedelta,
        offset: str | timedelta | None = None,
        closed: ClosedInterval = "right",
        group_by: IntoExpr | Iterable[IntoExpr] | None = None,
    ) -> RollingGroupByNameSpace:
        return RollingGroupByNameSpace(
            self,
            index_column,
            period=period,
            offset=offset,
            closed=closed,
            group_by=group_by,
        )

    def apply(self, func: Callable[[DataFrame], DataFrame]) -> Self:
        return self.pipe(Apply(func))

    def concat(
        self,
        items: Sequence[Transformer],
        *,
        how: ConcatMethod = "vertical",
        rechunk: bool = False,
        parallel: bool = True,
    ) -> Self:
        return self.pipe(Concat(items, how=how, rechunk=rechunk, parallel=parallel))

    def const(self, data: DataFrame) -> Self:
        return self.pipe(Const(data))

    def echo(self) -> Self:
        return self.pipe(Echo())

    def replay(self) -> Self:
        return self.pipe(Replay())

    def side(self, transformer: Transformer) -> Self:
        return self.pipe(Side(transformer))

    def combine(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        n: int,
        *,
        delimiter: str = "_",
        suffix: str = "_comb",
    ) -> Self:
        return self.pipe(Combine(columns, n, delimiter=delimiter, suffix=suffix))

    def discretize(
        self,
        exprs: IntoExpr | Iterable[IntoExpr],
        *more_exprs: IntoExpr | Iterable[IntoExpr],
        quantiles: Sequence[float] | int,
        labels: Sequence[str] | None = None,
        left_closed: bool = False,
        allow_duplicates: bool = False,
        suffix: str = "_disc",
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

    def join_agg(
        self,
        by: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *aggs: IntoExpr | Iterable[IntoExpr],
        how: JoinStrategy = "left",
        suffix: str = "_agg",
    ) -> Self:
        return self.pipe(JoinAgg(by, *aggs, how=how, suffix=suffix))

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
        step = LabelEncode(
            columns,
            *more_columns,
            orders=orders,
            maintain_order=maintain_order,
            suffix=suffix,
        )
        if inverse_mapping is None:
            return self.pipe(step)
        else:
            return LabelEncodeInverseContext(self, step, inverse_mapping)

    @overload
    def boxcox(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
    ) -> Self: ...

    @overload
    def boxcox(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
        inverse_mapping: Mapping[str, str] | None,
    ) -> PowerTransformInverseContext: ...

    def boxcox(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
        inverse_mapping: Mapping[str, str] | None = None,
    ) -> Self | PowerTransformInverseContext:
        step = BoxCoxTransform(columns, *more_columns, by=by, suffix=suffix)
        if inverse_mapping is None:
            return self.pipe(step)
        else:
            return PowerTransformInverseContext(self, step, inverse_mapping)

    @overload
    def yeojohnson(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
    ) -> Self: ...

    @overload
    def yeojohnson(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
        inverse_mapping: Mapping[str, str] | None,
    ) -> PowerTransformInverseContext: ...

    def yeojohnson(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        suffix: str = "",
        inverse_mapping: Mapping[str, str] | None = None,
    ) -> Self | PowerTransformInverseContext:
        step = YeoJohnsonTransform(columns, *more_columns, by=by, suffix=suffix)
        if inverse_mapping is None:
            return self.pipe(step)
        else:
            return PowerTransformInverseContext(self, step, inverse_mapping)

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
        step = MinMaxScale(columns, *more_columns, by=by, suffix=suffix)
        if inverse_mapping is None:
            return self.pipe(step)
        else:
            return ScaleInverseContext(self, step, inverse_mapping)

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
        step = StandardScale(columns, *more_columns, by=by, suffix=suffix)
        if inverse_mapping is None:
            return self.pipe(step)
        else:
            return ScaleInverseContext(self, step, inverse_mapping)

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
        step = RobustScale(
            columns, *more_columns, by=by, quantile_range=quantile_range, suffix=suffix
        )
        if inverse_mapping is None:
            return self.pipe(step)
        else:
            return ScaleInverseContext(self, step, inverse_mapping)

    # --- START INSERTION MARKER IN Pipeline

    def approx_n_unique(self) -> Self:
        return self.pipe(GetAttr("approx_n_unique", None))

    def bottom_k(
        self,
        k: int,
        by: IntoExpr | Iterable[IntoExpr],
        reverse: bool | Sequence[bool] = False,
    ) -> Self:
        return self.pipe(GetAttr("bottom_k", None, k, by=by, reverse=reverse))

    def cast(
        self,
        dtypes: Mapping[
            ColumnNameOrSelector | PolarsDataType, PolarsDataType | PythonDataType
        ]
        | PolarsDataType,
        strict: bool = True,
    ) -> Self:
        return self.pipe(GetAttr("cast", None, dtypes, strict=strict))

    def clear(self, n: int = 0) -> Self:
        return self.pipe(GetAttr("clear", None, n))

    def clone(self) -> Self:
        return self.pipe(GetAttr("clone", None))

    def corr(self, **kwargs: Any) -> Self:
        return self.pipe(GetAttr("corr", None, **kwargs))

    def count(self) -> Self:
        return self.pipe(GetAttr("count", None))

    def describe(
        self,
        percentiles: Sequence[float] | float | None = (0.25, 0.5, 0.75),
        interpolation: QuantileMethod = "nearest",
    ) -> Self:
        return self.pipe(
            GetAttr("describe", None, percentiles, interpolation=interpolation)
        )

    def deserialize(
        self, source: str | Path | IOBase, format: SerializationFormat = "binary"
    ) -> Self:
        return self.pipe(GetAttr("deserialize", None, source, format=format))

    def drop(
        self,
        *columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        strict: bool = True,
    ) -> Self:
        return self.pipe(GetAttr("drop", None, *columns, strict=strict))

    def drop_nans(
        self,
        subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
    ) -> Self:
        return self.pipe(GetAttr("drop_nans", None, subset))

    def drop_nulls(
        self,
        subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
    ) -> Self:
        return self.pipe(GetAttr("drop_nulls", None, subset))

    def explode(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
        empty_as_null: bool = True,
        keep_nulls: bool = True,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "explode",
                None,
                columns,
                *more_columns,
                empty_as_null=empty_as_null,
                keep_nulls=keep_nulls,
            )
        )

    def extend(self, other: DataFrame | Transformer) -> Self:
        return self.pipe(GetAttr("extend", None, other))

    def fill_nan(self, value: Expr | int | float | None) -> Self:
        return self.pipe(GetAttr("fill_nan", None, value))

    def fill_null(
        self,
        value: Any | Expr | None = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
        matches_supertype: bool = True,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "fill_null",
                None,
                value,
                strategy,
                limit,
                matches_supertype=matches_supertype,
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
        return self.pipe(GetAttr("filter", None, *predicates, **constraints))

    def gather_every(self, n: int, offset: int = 0) -> Self:
        return self.pipe(GetAttr("gather_every", None, n, offset))

    def head(self, n: int = 5) -> Self:
        return self.pipe(GetAttr("head", None, n))

    def hstack(self, columns: list[Series] | DataFrame, in_place: bool = False) -> Self:
        return self.pipe(GetAttr("hstack", None, columns, in_place=in_place))

    def insert_column(self, index: int, column: IntoExprColumn) -> Self:
        return self.pipe(GetAttr("insert_column", None, index, column))

    def interpolate(self) -> Self:
        return self.pipe(GetAttr("interpolate", None))

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
                None,
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
                None,
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
        return self.pipe(GetAttr("join_where", None, other, *predicates, suffix=suffix))

    def limit(self, n: int = 5) -> Self:
        return self.pipe(GetAttr("limit", None, n))

    def map_rows(
        self,
        function: Callable[[tuple[Any, ...]], Any],
        return_dtype: PolarsDataType | None = None,
        inference_size: int = 256,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "map_rows", None, function, return_dtype, inference_size=inference_size
            )
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
                None,
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
        return self.pipe(GetAttr("max", None))

    def mean(self) -> Self:
        return self.pipe(GetAttr("mean", None))

    def median(self) -> Self:
        return self.pipe(GetAttr("median", None))

    def melt(
        self,
        id_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        value_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        variable_name: str | None = None,
        value_name: str | None = None,
    ) -> Self:
        return self.pipe(
            GetAttr("melt", None, id_vars, value_vars, variable_name, value_name)
        )

    def merge_sorted(self, other: DataFrame | Transformer, key: str) -> Self:
        return self.pipe(GetAttr("merge_sorted", None, other, key))

    def min(self) -> Self:
        return self.pipe(GetAttr("min", None))

    def null_count(self) -> Self:
        return self.pipe(GetAttr("null_count", None))

    def pivot(
        self,
        on: ColumnNameOrSelector | Sequence[ColumnNameOrSelector],
        on_columns: Sequence[Any] | pl.Series | pl.DataFrame | None = None,
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
                None,
                on,
                on_columns,
                index=index,
                values=values,
                aggregate_function=aggregate_function,
                maintain_order=maintain_order,
                sort_columns=sort_columns,
                separator=separator,
            )
        )

    def product(self) -> Self:
        return self.pipe(GetAttr("product", None))

    def quantile(
        self, quantile: float, interpolation: QuantileMethod = "nearest"
    ) -> Self:
        return self.pipe(GetAttr("quantile", None, quantile, interpolation))

    def rechunk(self) -> Self:
        return self.pipe(GetAttr("rechunk", None))

    def remove(
        self,
        *predicates: IntoExprColumn
        | Iterable[IntoExprColumn]
        | bool
        | list[bool]
        | np.ndarray[Any, Any],
        **constraints: Any,
    ) -> Self:
        return self.pipe(GetAttr("remove", None, *predicates, **constraints))

    def rename(
        self, mapping: Mapping[str, str] | Callable[[str], str], strict: bool = True
    ) -> Self:
        return self.pipe(GetAttr("rename", None, mapping, strict=strict))

    def replace_column(self, index: int, column: Series) -> Self:
        return self.pipe(GetAttr("replace_column", None, index, column))

    def reverse(self) -> Self:
        return self.pipe(GetAttr("reverse", None))

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
                None,
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
        return self.pipe(GetAttr("select", None, *exprs, **named_exprs))

    def select_seq(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.pipe(GetAttr("select_seq", None, *exprs, **named_exprs))

    def set_sorted(self, column: str, descending: bool = False) -> Self:
        return self.pipe(GetAttr("set_sorted", None, column, descending=descending))

    def shift(self, n: int = 1, fill_value: IntoExpr | None = None) -> Self:
        return self.pipe(GetAttr("shift", None, n, fill_value=fill_value))

    def shrink_to_fit(self, in_place: bool = False) -> Self:
        return self.pipe(GetAttr("shrink_to_fit", None, in_place=in_place))

    def slice(self, offset: int, length: int | None = None) -> Self:
        return self.pipe(GetAttr("slice", None, offset, length))

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
                None,
                by,
                *more_by,
                descending=descending,
                nulls_last=nulls_last,
                multithreaded=multithreaded,
                maintain_order=maintain_order,
            )
        )

    def sql(self, query: str, table_name: str = "self") -> Self:
        return self.pipe(GetAttr("sql", None, query, table_name=table_name))

    def std(self, ddof: int = 1) -> Self:
        return self.pipe(GetAttr("std", None, ddof))

    def sum(self) -> Self:
        return self.pipe(GetAttr("sum", None))

    def tail(self, n: int = 5) -> Self:
        return self.pipe(GetAttr("tail", None, n))

    def to_dummies(
        self,
        columns: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        separator: str = "_",
        drop_first: bool = False,
        drop_nulls: bool = False,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "to_dummies",
                None,
                columns,
                separator=separator,
                drop_first=drop_first,
                drop_nulls=drop_nulls,
            )
        )

    def top_k(
        self,
        k: int,
        by: IntoExpr | Iterable[IntoExpr],
        reverse: bool | Sequence[bool] = False,
    ) -> Self:
        return self.pipe(GetAttr("top_k", None, k, by=by, reverse=reverse))

    def transpose(
        self,
        include_header: bool = False,
        header_name: str = "column",
        column_names: str | Iterable[str] | None = None,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "transpose",
                None,
                include_header=include_header,
                header_name=header_name,
                column_names=column_names,
            )
        )

    def unique(
        self,
        subset: IntoExpr | Collection[IntoExpr] | None = None,
        keep: UniqueKeepStrategy = "any",
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            GetAttr("unique", None, subset, keep=keep, maintain_order=maintain_order)
        )

    def unnest(
        self,
        columns: ColumnNameOrSelector | Collection[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
        separator: str | None = None,
    ) -> Self:
        return self.pipe(
            GetAttr("unnest", None, columns, *more_columns, separator=separator)
        )

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
                None,
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
                "unstack",
                None,
                step=step,
                how=how,
                columns=columns,
                fill_values=fill_values,
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
                None,
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
                None,
                time_column,
                every=every,
                group_by=group_by,
                maintain_order=maintain_order,
            )
        )

    def var(self, ddof: int = 1) -> Self:
        return self.pipe(GetAttr("var", None, ddof))

    def vstack(self, other: DataFrame | Transformer, in_place: bool = False) -> Self:
        return self.pipe(GetAttr("vstack", None, other, in_place=in_place))

    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.pipe(GetAttr("with_columns", None, *exprs, **named_exprs))

    def with_columns_seq(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.pipe(GetAttr("with_columns_seq", None, *exprs, **named_exprs))

    def with_row_count(self, name: str = "row_nr", offset: int = 0) -> Self:
        return self.pipe(GetAttr("with_row_count", None, name, offset))

    def with_row_index(self, name: str = "index", offset: int = 0) -> Self:
        return self.pipe(GetAttr("with_row_index", None, name, offset))

    def write_avro(
        self,
        file: str | Path | IO[bytes],
        compression: AvroCompression = "uncompressed",
        name: str = "",
    ) -> Self:
        return self.pipe(GetAttr("write_avro", None, file, compression, name))

    def write_clipboard(self, separator: str = "\t", **kwargs: Any) -> Self:
        return self.pipe(
            GetAttr("write_clipboard", None, separator=separator, **kwargs)
        )

    def write_csv(
        self,
        file: str | Path | IO[str] | IO[bytes] | None = None,
        include_bom: bool = False,
        include_header: bool = True,
        separator: str = ",",
        line_terminator: str = "\n",
        quote_char: str = '"',
        batch_size: int = 1024,
        datetime_format: str | None = None,
        date_format: str | None = None,
        time_format: str | None = None,
        float_scientific: bool | None = None,
        float_precision: int | None = None,
        decimal_comma: bool = False,
        null_value: str | None = None,
        quote_style: CsvQuoteStyle | None = None,
        storage_options: dict[str, Any] | None = None,
        credential_provider: CredentialProviderFunction
        | Literal["auto"]
        | None = "auto",
        retries: int = 2,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "write_csv",
                None,
                file,
                include_bom=include_bom,
                include_header=include_header,
                separator=separator,
                line_terminator=line_terminator,
                quote_char=quote_char,
                batch_size=batch_size,
                datetime_format=datetime_format,
                date_format=date_format,
                time_format=time_format,
                float_scientific=float_scientific,
                float_precision=float_precision,
                decimal_comma=decimal_comma,
                null_value=null_value,
                quote_style=quote_style,
                storage_options=storage_options,
                credential_provider=credential_provider,
                retries=retries,
            )
        )

    def write_database(
        self,
        table_name: str,
        connection: ConnectionOrCursor | str,
        if_table_exists: DbWriteMode = "fail",
        engine: DbWriteEngine | None = None,
        engine_options: dict[str, Any] | None = None,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "write_database",
                None,
                table_name,
                connection,
                if_table_exists=if_table_exists,
                engine=engine,
                engine_options=engine_options,
            )
        )

    def write_delta(
        self,
        target: str | Path | deltalake.DeltaTable,
        mode: Literal["error", "append", "overwrite", "ignore", "merge"] = "error",
        overwrite_schema: bool | None = None,
        storage_options: dict[str, str] | None = None,
        credential_provider: CredentialProviderFunction
        | Literal["auto"]
        | None = "auto",
        delta_write_options: dict[str, Any] | None = None,
        delta_merge_options: dict[str, Any] | None = None,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "write_delta",
                None,
                target,
                mode=mode,
                overwrite_schema=overwrite_schema,
                storage_options=storage_options,
                credential_provider=credential_provider,
                delta_write_options=delta_write_options,
                delta_merge_options=delta_merge_options,
            )
        )

    def write_excel(
        self,
        workbook: str | Workbook | IO[bytes] | Path | None = None,
        worksheet: str | Worksheet | None = None,
        position: tuple[int, int] | str = "A1",
        table_style: str | dict[str, Any] | None = None,
        table_name: str | None = None,
        column_formats: ColumnFormatDict | None = None,
        dtype_formats: dict[OneOrMoreDataTypes, str] | None = None,
        conditional_formats: ConditionalFormatDict | None = None,
        header_format: dict[str, Any] | None = None,
        column_totals: ColumnTotalsDefinition | None = None,
        column_widths: ColumnWidthsDefinition | None = None,
        row_totals: RowTotalsDefinition | None = None,
        row_heights: dict[int | tuple[int, ...], int] | int | None = None,
        sparklines: dict[str, Sequence[str] | dict[str, Any]] | None = None,
        formulas: dict[str, str | dict[str, str]] | None = None,
        float_precision: int = 3,
        include_header: bool = True,
        autofilter: bool = True,
        autofit: bool = False,
        hidden_columns: Sequence[str] | SelectorType | None = None,
        hide_gridlines: bool = False,
        sheet_zoom: int | None = None,
        freeze_panes: str
        | tuple[int, int]
        | tuple[str, int, int]
        | tuple[int, int, int, int]
        | None = None,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "write_excel",
                None,
                workbook,
                worksheet,
                position=position,
                table_style=table_style,
                table_name=table_name,
                column_formats=column_formats,
                dtype_formats=dtype_formats,
                conditional_formats=conditional_formats,
                header_format=header_format,
                column_totals=column_totals,
                column_widths=column_widths,
                row_totals=row_totals,
                row_heights=row_heights,
                sparklines=sparklines,
                formulas=formulas,
                float_precision=float_precision,
                include_header=include_header,
                autofilter=autofilter,
                autofit=autofit,
                hidden_columns=hidden_columns,
                hide_gridlines=hide_gridlines,
                sheet_zoom=sheet_zoom,
                freeze_panes=freeze_panes,
            )
        )

    def write_ipc(
        self,
        file: str | Path | IO[bytes] | None,
        compression: IpcCompression = "uncompressed",
        compat_level: CompatLevel | None = None,
        storage_options: dict[str, Any] | None = None,
        credential_provider: CredentialProviderFunction
        | Literal["auto"]
        | None = "auto",
        retries: int = 2,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "write_ipc",
                None,
                file,
                compression=compression,
                compat_level=compat_level,
                storage_options=storage_options,
                credential_provider=credential_provider,
                retries=retries,
            )
        )

    def write_ipc_stream(
        self,
        file: str | Path | IO[bytes] | None,
        compression: IpcCompression = "uncompressed",
        compat_level: CompatLevel | None = None,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "write_ipc_stream",
                None,
                file,
                compression=compression,
                compat_level=compat_level,
            )
        )

    def write_json(self, file: IOBase | str | Path | None = None) -> Self:
        return self.pipe(GetAttr("write_json", None, file))

    def write_ndjson(
        self, file: str | Path | IO[bytes] | IO[str] | None = None
    ) -> Self:
        return self.pipe(GetAttr("write_ndjson", None, file))

    def write_parquet(
        self,
        file: str | Path | IO[bytes],
        compression: ParquetCompression = "zstd",
        compression_level: int | None = None,
        statistics: bool | str | dict[str, bool] = True,
        row_group_size: int | None = None,
        data_page_size: int | None = None,
        use_pyarrow: bool = False,
        pyarrow_options: dict[str, Any] | None = None,
        partition_by: str | Sequence[str] | None = None,
        partition_chunk_size_bytes: int = 4294967296,
        storage_options: dict[str, Any] | None = None,
        credential_provider: CredentialProviderFunction
        | Literal["auto"]
        | None = "auto",
        retries: int = 2,
        metadata: ParquetMetadata | None = None,
        mkdir: bool = False,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "write_parquet",
                None,
                file,
                compression=compression,
                compression_level=compression_level,
                statistics=statistics,
                row_group_size=row_group_size,
                data_page_size=data_page_size,
                use_pyarrow=use_pyarrow,
                pyarrow_options=pyarrow_options,
                partition_by=partition_by,
                partition_chunk_size_bytes=partition_chunk_size_bytes,
                storage_options=storage_options,
                credential_provider=credential_provider,
                retries=retries,
                metadata=metadata,
                mkdir=mkdir,
            )
        )

    def read_avro(
        self,
        source: str | Path | IO[bytes] | bytes,
        columns: list[int] | list[str] | None = None,
        n_rows: int | None = None,
    ) -> Self:
        return self.pipe(
            GetAttr("read_avro", pl, source, columns=columns, n_rows=n_rows)
        )

    def read_clipboard(self, separator: str = "\t", **kwargs: Any) -> Self:
        return self.pipe(GetAttr("read_clipboard", pl, separator, **kwargs))

    def read_csv(
        self,
        source: str | Path | IO[str] | IO[bytes] | bytes,
        has_header: bool = True,
        columns: Sequence[int] | Sequence[str] | None = None,
        new_columns: Sequence[str] | None = None,
        separator: str = ",",
        comment_prefix: str | None = None,
        quote_char: str | None = '"',
        skip_rows: int = 0,
        skip_lines: int = 0,
        schema: SchemaDict | None = None,
        schema_overrides: Mapping[str, PolarsDataType]
        | Sequence[PolarsDataType]
        | None = None,
        null_values: str | Sequence[str] | dict[str, str] | None = None,
        missing_utf8_is_empty_string: bool = False,
        ignore_errors: bool = False,
        try_parse_dates: bool = False,
        n_threads: int | None = None,
        infer_schema: bool = True,
        infer_schema_length: int | None = 100,
        batch_size: int = 8192,
        n_rows: int | None = None,
        encoding: CsvEncoding | str = "utf8",
        low_memory: bool = False,
        rechunk: bool = False,
        use_pyarrow: bool = False,
        storage_options: dict[str, Any] | None = None,
        skip_rows_after_header: int = 0,
        row_index_name: str | None = None,
        row_index_offset: int = 0,
        sample_size: int = 1024,
        eol_char: str = "\n",
        raise_if_empty: bool = True,
        truncate_ragged_lines: bool = False,
        decimal_comma: bool = False,
        glob: bool = True,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "read_csv",
                pl,
                source,
                has_header=has_header,
                columns=columns,
                new_columns=new_columns,
                separator=separator,
                comment_prefix=comment_prefix,
                quote_char=quote_char,
                skip_rows=skip_rows,
                skip_lines=skip_lines,
                schema=schema,
                schema_overrides=schema_overrides,
                null_values=null_values,
                missing_utf8_is_empty_string=missing_utf8_is_empty_string,
                ignore_errors=ignore_errors,
                try_parse_dates=try_parse_dates,
                n_threads=n_threads,
                infer_schema=infer_schema,
                infer_schema_length=infer_schema_length,
                batch_size=batch_size,
                n_rows=n_rows,
                encoding=encoding,
                low_memory=low_memory,
                rechunk=rechunk,
                use_pyarrow=use_pyarrow,
                storage_options=storage_options,
                skip_rows_after_header=skip_rows_after_header,
                row_index_name=row_index_name,
                row_index_offset=row_index_offset,
                sample_size=sample_size,
                eol_char=eol_char,
                raise_if_empty=raise_if_empty,
                truncate_ragged_lines=truncate_ragged_lines,
                decimal_comma=decimal_comma,
                glob=glob,
            )
        )

    def read_database_uri(
        self,
        query: list[str] | str,
        uri: str,
        partition_on: str | None = None,
        partition_range: tuple[int, int] | None = None,
        partition_num: int | None = None,
        protocol: str | None = None,
        engine: DbReadEngine | None = None,
        schema_overrides: SchemaDict | None = None,
        execute_options: dict[str, Any] | None = None,
        pre_execution_query: str | list[str] | None = None,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "read_database_uri",
                pl,
                query,
                uri,
                partition_on=partition_on,
                partition_range=partition_range,
                partition_num=partition_num,
                protocol=protocol,
                engine=engine,
                schema_overrides=schema_overrides,
                execute_options=execute_options,
                pre_execution_query=pre_execution_query,
            )
        )

    def read_delta(
        self,
        source: str | Path | DeltaTable,
        version: int | str | datetime | None = None,
        columns: list[str] | None = None,
        rechunk: bool | None = None,
        storage_options: dict[str, Any] | None = None,
        credential_provider: CredentialProviderFunction
        | Literal["auto"]
        | None = "auto",
        delta_table_options: dict[str, Any] | None = None,
        use_pyarrow: bool = False,
        pyarrow_options: dict[str, Any] | None = None,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "read_delta",
                pl,
                source,
                version=version,
                columns=columns,
                rechunk=rechunk,
                storage_options=storage_options,
                credential_provider=credential_provider,
                delta_table_options=delta_table_options,
                use_pyarrow=use_pyarrow,
                pyarrow_options=pyarrow_options,
            )
        )

    def read_ipc(
        self,
        source: str | Path | IO[bytes] | bytes,
        columns: list[int] | list[str] | None = None,
        n_rows: int | None = None,
        use_pyarrow: bool = False,
        memory_map: bool = True,
        storage_options: dict[str, Any] | None = None,
        row_index_name: str | None = None,
        row_index_offset: int = 0,
        rechunk: bool = True,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "read_ipc",
                pl,
                source,
                columns=columns,
                n_rows=n_rows,
                use_pyarrow=use_pyarrow,
                memory_map=memory_map,
                storage_options=storage_options,
                row_index_name=row_index_name,
                row_index_offset=row_index_offset,
                rechunk=rechunk,
            )
        )

    def read_ipc_stream(
        self,
        source: str | Path | IO[bytes] | bytes,
        columns: list[int] | list[str] | None = None,
        n_rows: int | None = None,
        use_pyarrow: bool = False,
        storage_options: dict[str, Any] | None = None,
        row_index_name: str | None = None,
        row_index_offset: int = 0,
        rechunk: bool = True,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "read_ipc_stream",
                pl,
                source,
                columns=columns,
                n_rows=n_rows,
                use_pyarrow=use_pyarrow,
                storage_options=storage_options,
                row_index_name=row_index_name,
                row_index_offset=row_index_offset,
                rechunk=rechunk,
            )
        )

    def read_json(
        self,
        source: str | Path | IOBase | bytes,
        schema: SchemaDefinition | None = None,
        schema_overrides: SchemaDefinition | None = None,
        infer_schema_length: int | None = 100,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "read_json",
                pl,
                source,
                schema=schema,
                schema_overrides=schema_overrides,
                infer_schema_length=infer_schema_length,
            )
        )

    def read_ndjson(
        self,
        source: str
        | Path
        | IO[str]
        | IO[bytes]
        | bytes
        | list[str]
        | list[Path]
        | list[IO[str]]
        | list[IO[bytes]],
        schema: SchemaDefinition | None = None,
        schema_overrides: SchemaDefinition | None = None,
        infer_schema_length: int | None = 100,
        batch_size: int | None = 1024,
        n_rows: int | None = None,
        low_memory: bool = False,
        rechunk: bool = False,
        row_index_name: str | None = None,
        row_index_offset: int = 0,
        ignore_errors: bool = False,
        storage_options: dict[str, Any] | None = None,
        credential_provider: CredentialProviderFunction
        | Literal["auto"]
        | None = "auto",
        retries: int = 2,
        file_cache_ttl: int | None = None,
        include_file_paths: str | None = None,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "read_ndjson",
                pl,
                source,
                schema=schema,
                schema_overrides=schema_overrides,
                infer_schema_length=infer_schema_length,
                batch_size=batch_size,
                n_rows=n_rows,
                low_memory=low_memory,
                rechunk=rechunk,
                row_index_name=row_index_name,
                row_index_offset=row_index_offset,
                ignore_errors=ignore_errors,
                storage_options=storage_options,
                credential_provider=credential_provider,
                retries=retries,
                file_cache_ttl=file_cache_ttl,
                include_file_paths=include_file_paths,
            )
        )

    def read_parquet(
        self,
        source: FileSource,
        columns: list[int] | list[str] | None = None,
        n_rows: int | None = None,
        row_index_name: str | None = None,
        row_index_offset: int = 0,
        parallel: ParallelStrategy = "auto",
        use_statistics: bool = True,
        hive_partitioning: bool | None = None,
        glob: bool = True,
        schema: SchemaDict | None = None,
        hive_schema: SchemaDict | None = None,
        try_parse_hive_dates: bool = True,
        rechunk: bool = False,
        low_memory: bool = False,
        storage_options: dict[str, Any] | None = None,
        credential_provider: CredentialProviderFunction
        | Literal["auto"]
        | None = "auto",
        retries: int = 2,
        use_pyarrow: bool = False,
        pyarrow_options: dict[str, Any] | None = None,
        memory_map: bool = True,
        include_file_paths: str | None = None,
        missing_columns: Literal["insert", "raise"] = "raise",
        allow_missing_columns: bool | None = None,
    ) -> Self:
        return self.pipe(
            GetAttr(
                "read_parquet",
                pl,
                source,
                columns=columns,
                n_rows=n_rows,
                row_index_name=row_index_name,
                row_index_offset=row_index_offset,
                parallel=parallel,
                use_statistics=use_statistics,
                hive_partitioning=hive_partitioning,
                glob=glob,
                schema=schema,
                hive_schema=hive_schema,
                try_parse_hive_dates=try_parse_hive_dates,
                rechunk=rechunk,
                low_memory=low_memory,
                storage_options=storage_options,
                credential_provider=credential_provider,
                retries=retries,
                use_pyarrow=use_pyarrow,
                pyarrow_options=pyarrow_options,
                memory_map=memory_map,
                include_file_paths=include_file_paths,
                missing_columns=missing_columns,
                allow_missing_columns=allow_missing_columns,
            )
        )

    # --- END INSERTION MARKER IN Pipeline
