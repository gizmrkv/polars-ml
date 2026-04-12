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
)

import numpy as np
import polars as pl
from polars import DataFrame, Expr, LazyFrame, Schema

from polars_ml import LazyTransformer

from .basic import LazyApply, LazyConst, LazySide
from .getattr import LazyGetAttr
from .group_by_lazy import LazyGroupByNameSpace
from .pipeline_mixin import PipelineMixin

if TYPE_CHECKING:
    from deltalake import DeltaTable
    from polars._typing import (
        AsofJoinStrategy,
        ClosedInterval,
        ColumnMapping,
        ColumnNameOrSelector,
        CsvEncoding,
        DefaultFieldValues,
        DeletionFiles,
        FileSource,
        FillNullStrategy,
        IntoExpr,
        IntoExprColumn,
        JoinStrategy,
        JoinValidation,
        Label,
        MaintainOrderJoin,
        ParallelStrategy,
        PivotAgg,
        PolarsDataType,
        PythonDataType,
        QuantileMethod,
        SchemaDefinition,
        SchemaDict,
        SerializationFormat,
        StartBy,
        StorageOptionsDict,
        UniqueKeepStrategy,
    )
    from polars.io.cloud import CredentialProviderFunction
    from polars.io.scan_options import ScanCastOptions


class LazyPipeline(LazyTransformer, PipelineMixin):
    def __init__(self, *steps: LazyTransformer) -> None:
        self._steps = list(steps)

    def pipe(self, step: LazyTransformer) -> Self:
        self._steps.append(step)
        return self

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        for i, step in enumerate(self._steps):
            if i < len(self._steps) - 1:
                data = step.fit_transform(data, **more_data)
                more_data = {
                    k: step.collect().transform(v) for k, v in more_data.items()
                }
            else:
                step.fit(data, **more_data)

        return self

    def fit_transform(
        self, data: pl.DataFrame, **more_data: pl.DataFrame
    ) -> pl.DataFrame:
        for i, step in enumerate(self._steps):
            data = step.fit_transform(data, **more_data)
            if i < len(self._steps) - 1:
                more_data = {
                    k: step.collect().transform(v) for k, v in more_data.items()
                }

        return data

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        for step in self._steps:
            data = step.transform(data)
        return data

    def apply(self, func: Callable[[pl.LazyFrame], pl.LazyFrame]) -> Self:
        return self.pipe(LazyApply(func))

    def const(self, data: pl.LazyFrame) -> Self:
        return self.pipe(LazyConst(data))

    def side(self, transformer: LazyTransformer) -> Self:
        return self.pipe(LazySide(transformer))

    def group_by(
        self,
        *by: IntoExpr | Iterable[IntoExpr],
        maintain_order: bool = False,
        **named_by: IntoExpr,
    ) -> LazyGroupByNameSpace:
        return LazyGroupByNameSpace(
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
    ) -> LazyGroupByNameSpace:
        return LazyGroupByNameSpace(
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

    def rolling(
        self,
        index_column: IntoExpr,
        *,
        period: str | timedelta,
        offset: str | timedelta | None = None,
        closed: ClosedInterval = "right",
        group_by: IntoExpr | Iterable[IntoExpr] | None = None,
    ) -> LazyGroupByNameSpace:
        return LazyGroupByNameSpace(
            self,
            "rolling",
            index_column,
            period=period,
            offset=offset,
            closed=closed,
            group_by=group_by,
        )

    # --- START INSERTION MARKER IN LazyPipeline

    def approx_n_unique(self) -> Self:
        return self.pipe(LazyGetAttr("approx_n_unique", None))

    def bottom_k(
        self,
        k: int,
        by: IntoExpr | Iterable[IntoExpr],
        reverse: bool | Sequence[bool] = False,
    ) -> Self:
        return self.pipe(LazyGetAttr("bottom_k", None, k, by=by, reverse=reverse))

    def cache(self) -> Self:
        return self.pipe(LazyGetAttr("cache", None))

    def cast(
        self,
        dtypes: Mapping[
            ColumnNameOrSelector | PolarsDataType, PolarsDataType | PythonDataType
        ]
        | PolarsDataType
        | pl.DataTypeExpr
        | Schema,
        strict: bool = True,
    ) -> Self:
        return self.pipe(LazyGetAttr("cast", None, dtypes, strict=strict))

    def clear(self, n: int = 0) -> Self:
        return self.pipe(LazyGetAttr("clear", None, n))

    def clone(self) -> Self:
        return self.pipe(LazyGetAttr("clone", None))

    def count(self) -> Self:
        return self.pipe(LazyGetAttr("count", None))

    def deserialize(
        self,
        source: str | bytes | Path | IOBase,
        format: SerializationFormat = "binary",
    ) -> Self:
        return self.pipe(LazyGetAttr("deserialize", None, source, format=format))

    def drop(
        self,
        *columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        strict: bool = True,
    ) -> Self:
        return self.pipe(LazyGetAttr("drop", None, *columns, strict=strict))

    def drop_nans(
        self,
        subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
    ) -> Self:
        return self.pipe(LazyGetAttr("drop_nans", None, subset))

    def drop_nulls(
        self,
        subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
    ) -> Self:
        return self.pipe(LazyGetAttr("drop_nulls", None, subset))

    def explode(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
        empty_as_null: bool = True,
        keep_nulls: bool = True,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "explode",
                None,
                columns,
                *more_columns,
                empty_as_null=empty_as_null,
                keep_nulls=keep_nulls,
            )
        )

    def fill_nan(self, value: int | float | Expr | None) -> Self:
        return self.pipe(LazyGetAttr("fill_nan", None, value))

    def fill_null(
        self,
        value: Any | Expr | None = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
        matches_supertype: bool = True,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
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
        return self.pipe(LazyGetAttr("filter", None, *predicates, **constraints))

    def first(self) -> Self:
        return self.pipe(LazyGetAttr("first", None))

    def gather_every(self, n: int, offset: int = 0) -> Self:
        return self.pipe(LazyGetAttr("gather_every", None, n, offset))

    def head(self, n: int = 5) -> Self:
        return self.pipe(LazyGetAttr("head", None, n))

    def inspect(self, fmt: str = "{}") -> Self:
        return self.pipe(LazyGetAttr("inspect", None, fmt))

    def interpolate(self) -> Self:
        return self.pipe(LazyGetAttr("interpolate", None))

    def join(
        self,
        other: pl.LazyFrame | LazyTransformer,
        on: str | Expr | Sequence[str | Expr] | None = None,
        how: JoinStrategy = "inner",
        left_on: str | Expr | Sequence[str | Expr] | None = None,
        right_on: str | Expr | Sequence[str | Expr] | None = None,
        suffix: str = "_right",
        validate: JoinValidation = "m:m",
        nulls_equal: bool = False,
        coalesce: bool | None = None,
        maintain_order: MaintainOrderJoin | None = None,
        allow_parallel: bool = True,
        force_parallel: bool = False,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
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
                allow_parallel=allow_parallel,
                force_parallel=force_parallel,
            )
        )

    def join_asof(
        self,
        other: pl.LazyFrame | LazyTransformer,
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
            LazyGetAttr(
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
        other: pl.LazyFrame | LazyTransformer,
        *predicates: Expr | Iterable[Expr],
        suffix: str = "_right",
    ) -> Self:
        return self.pipe(
            LazyGetAttr("join_where", None, other, *predicates, suffix=suffix)
        )

    def last(self) -> Self:
        return self.pipe(LazyGetAttr("last", None))

    def lazy(self) -> Self:
        return self.pipe(LazyGetAttr("lazy", None))

    def limit(self, n: int = 5) -> Self:
        return self.pipe(LazyGetAttr("limit", None, n))

    def map_batches(
        self,
        function: Callable[[DataFrame], DataFrame],
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        slice_pushdown: bool = True,
        no_optimizations: bool = False,
        schema: None | SchemaDict = None,
        validate_output_schema: bool = True,
        streamable: bool = False,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "map_batches",
                None,
                function,
                predicate_pushdown=predicate_pushdown,
                projection_pushdown=projection_pushdown,
                slice_pushdown=slice_pushdown,
                no_optimizations=no_optimizations,
                schema=schema,
                validate_output_schema=validate_output_schema,
                streamable=streamable,
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
            LazyGetAttr(
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
        return self.pipe(LazyGetAttr("max", None))

    def mean(self) -> Self:
        return self.pipe(LazyGetAttr("mean", None))

    def median(self) -> Self:
        return self.pipe(LazyGetAttr("median", None))

    def melt(
        self,
        id_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        value_vars: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        variable_name: str | None = None,
        value_name: str | None = None,
        streamable: bool = True,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "melt",
                None,
                id_vars,
                value_vars,
                variable_name,
                value_name,
                streamable=streamable,
            )
        )

    def merge_sorted(self, other: pl.LazyFrame | LazyTransformer, key: str) -> Self:
        return self.pipe(LazyGetAttr("merge_sorted", None, other, key))

    def min(self) -> Self:
        return self.pipe(LazyGetAttr("min", None))

    def null_count(self) -> Self:
        return self.pipe(LazyGetAttr("null_count", None))

    def pipe_with_schema(
        self, function: Callable[[LazyFrame, Schema], LazyFrame]
    ) -> Self:
        return self.pipe(LazyGetAttr("pipe_with_schema", None, function))

    def pivot(
        self,
        on: ColumnNameOrSelector | Sequence[ColumnNameOrSelector],
        on_columns: Sequence[Any] | pl.Series | pl.DataFrame,
        index: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        values: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        aggregate_function: PivotAgg | Expr | None = None,
        maintain_order: bool = False,
        separator: str = "_",
        column_naming: Literal["auto", "combine"] = "auto",
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "pivot",
                None,
                on,
                on_columns,
                index=index,
                values=values,
                aggregate_function=aggregate_function,
                maintain_order=maintain_order,
                separator=separator,
                column_naming=column_naming,
            )
        )

    def quantile(
        self, quantile: float | Expr, interpolation: QuantileMethod = "nearest"
    ) -> Self:
        return self.pipe(LazyGetAttr("quantile", None, quantile, interpolation))

    def remove(
        self,
        *predicates: IntoExprColumn
        | Iterable[IntoExprColumn]
        | bool
        | list[bool]
        | np.ndarray[Any, Any],
        **constraints: Any,
    ) -> Self:
        return self.pipe(LazyGetAttr("remove", None, *predicates, **constraints))

    def rename(
        self, mapping: Mapping[str, str] | Callable[[str], str], strict: bool = True
    ) -> Self:
        return self.pipe(LazyGetAttr("rename", None, mapping, strict=strict))

    def reverse(self) -> Self:
        return self.pipe(LazyGetAttr("reverse", None))

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.pipe(LazyGetAttr("select", None, *exprs, **named_exprs))

    def select_seq(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.pipe(LazyGetAttr("select_seq", None, *exprs, **named_exprs))

    def set_sorted(
        self,
        column: str | list[str],
        *more_columns: str,
        descending: bool | list[bool] = False,
        nulls_last: bool | list[bool] = False,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "set_sorted",
                None,
                column,
                *more_columns,
                descending=descending,
                nulls_last=nulls_last,
            )
        )

    def shift(
        self, n: int | IntoExprColumn = 1, fill_value: IntoExpr | None = None
    ) -> Self:
        return self.pipe(LazyGetAttr("shift", None, n, fill_value=fill_value))

    def slice(self, offset: int, length: int | None = None) -> Self:
        return self.pipe(LazyGetAttr("slice", None, offset, length))

    def sort(
        self,
        by: IntoExpr | Iterable[IntoExpr],
        *more_by: IntoExpr,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool | Sequence[bool] = False,
        maintain_order: bool = False,
        multithreaded: bool = True,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "sort",
                None,
                by,
                *more_by,
                descending=descending,
                nulls_last=nulls_last,
                maintain_order=maintain_order,
                multithreaded=multithreaded,
            )
        )

    def sql(self, query: str, table_name: str = "self") -> Self:
        return self.pipe(LazyGetAttr("sql", None, query, table_name=table_name))

    def std(self, ddof: int = 1) -> Self:
        return self.pipe(LazyGetAttr("std", None, ddof))

    def sum(self) -> Self:
        return self.pipe(LazyGetAttr("sum", None))

    def tail(self, n: int = 5) -> Self:
        return self.pipe(LazyGetAttr("tail", None, n))

    def top_k(
        self,
        k: int,
        by: IntoExpr | Iterable[IntoExpr],
        reverse: bool | Sequence[bool] = False,
    ) -> Self:
        return self.pipe(LazyGetAttr("top_k", None, k, by=by, reverse=reverse))

    def unique(
        self,
        subset: IntoExpr | Collection[IntoExpr] | None = None,
        keep: UniqueKeepStrategy = "any",
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "unique", None, subset, keep=keep, maintain_order=maintain_order
            )
        )

    def unnest(
        self,
        columns: ColumnNameOrSelector | Collection[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
        separator: str | None = None,
    ) -> Self:
        return self.pipe(
            LazyGetAttr("unnest", None, columns, *more_columns, separator=separator)
        )

    def unpivot(
        self,
        on: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        index: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        variable_name: str | None = None,
        value_name: str | None = None,
        streamable: bool = True,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "unpivot",
                None,
                on,
                index=index,
                variable_name=variable_name,
                value_name=value_name,
                streamable=streamable,
            )
        )

    def update(
        self,
        other: pl.LazyFrame | LazyTransformer,
        on: str | Sequence[str] | None = None,
        how: Literal["left", "inner", "full"] = "left",
        left_on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        include_nulls: bool = False,
        maintain_order: MaintainOrderJoin | None = "left",
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
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

    def var(self, ddof: int = 1) -> Self:
        return self.pipe(LazyGetAttr("var", None, ddof))

    def with_columns(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.pipe(LazyGetAttr("with_columns", None, *exprs, **named_exprs))

    def with_columns_seq(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.pipe(LazyGetAttr("with_columns_seq", None, *exprs, **named_exprs))

    def with_context(self, other: Self | list[Self]) -> Self:
        return self.pipe(LazyGetAttr("with_context", None, other))

    def with_row_count(self, name: str = "row_nr", offset: int = 0) -> Self:
        return self.pipe(LazyGetAttr("with_row_count", None, name, offset))

    def with_row_index(self, name: str = "index", offset: int = 0) -> Self:
        return self.pipe(LazyGetAttr("with_row_index", None, name, offset))

    def scan_csv(
        self,
        source: str
        | Path
        | IO[str]
        | IO[bytes]
        | bytes
        | list[str]
        | list[Path]
        | list[IO[str]]
        | list[IO[bytes]]
        | list[bytes],
        has_header: bool = True,
        separator: str = ",",
        comment_prefix: str | None = None,
        quote_char: str | None = '"',
        skip_rows: int = 0,
        skip_lines: int = 0,
        schema: SchemaDict | None = None,
        schema_overrides: SchemaDict | Sequence[PolarsDataType] | None = None,
        null_values: str | Sequence[str] | dict[str, str] | None = None,
        missing_utf8_is_empty_string: bool = False,
        ignore_errors: bool = False,
        cache: bool | None = None,
        with_column_names: Callable[[list[str]], list[str]] | None = None,
        infer_schema: bool = True,
        infer_schema_length: int | None = 100,
        n_rows: int | None = None,
        encoding: CsvEncoding = "utf8",
        low_memory: bool = False,
        rechunk: bool = False,
        skip_rows_after_header: int = 0,
        row_index_name: str | None = None,
        row_index_offset: int = 0,
        try_parse_dates: bool = False,
        eol_char: str = "\n",
        new_columns: Sequence[str] | None = None,
        raise_if_empty: bool = True,
        truncate_ragged_lines: bool = False,
        decimal_comma: bool = False,
        glob: bool = True,
        storage_options: StorageOptionsDict | None = None,
        credential_provider: CredentialProviderFunction
        | Literal["auto"]
        | None = "auto",
        retries: int | None = None,
        file_cache_ttl: int | None = None,
        include_file_paths: str | None = None,
        missing_columns: Literal["insert", "raise"] | None = None,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "scan_csv",
                pl,
                source,
                has_header=has_header,
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
                cache=cache,
                with_column_names=with_column_names,
                infer_schema=infer_schema,
                infer_schema_length=infer_schema_length,
                n_rows=n_rows,
                encoding=encoding,
                low_memory=low_memory,
                rechunk=rechunk,
                skip_rows_after_header=skip_rows_after_header,
                row_index_name=row_index_name,
                row_index_offset=row_index_offset,
                try_parse_dates=try_parse_dates,
                eol_char=eol_char,
                new_columns=new_columns,
                raise_if_empty=raise_if_empty,
                truncate_ragged_lines=truncate_ragged_lines,
                decimal_comma=decimal_comma,
                glob=glob,
                storage_options=storage_options,
                credential_provider=credential_provider,
                retries=retries,
                file_cache_ttl=file_cache_ttl,
                include_file_paths=include_file_paths,
                missing_columns=missing_columns,
            )
        )

    def scan_delta(
        self,
        source: str | Path | DeltaTable,
        version: int | str | datetime | None = None,
        storage_options: StorageOptionsDict | None = None,
        credential_provider: CredentialProviderFunction
        | Literal["auto"]
        | None = "auto",
        delta_table_options: dict[str, Any] | None = None,
        use_pyarrow: bool = False,
        pyarrow_options: dict[str, Any] | None = None,
        rechunk: bool | None = None,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "scan_delta",
                pl,
                source,
                version=version,
                storage_options=storage_options,
                credential_provider=credential_provider,
                delta_table_options=delta_table_options,
                use_pyarrow=use_pyarrow,
                pyarrow_options=pyarrow_options,
                rechunk=rechunk,
            )
        )

    def scan_ipc(
        self,
        source: str
        | Path
        | IO[bytes]
        | bytes
        | list[str]
        | list[Path]
        | list[IO[bytes]]
        | list[bytes],
        n_rows: int | None = None,
        cache: bool = True,
        rechunk: bool = False,
        row_index_name: str | None = None,
        row_index_offset: int = 0,
        glob: bool = True,
        storage_options: StorageOptionsDict | None = None,
        credential_provider: CredentialProviderFunction
        | Literal["auto"]
        | None = "auto",
        memory_map: bool = True,
        retries: int | None = None,
        file_cache_ttl: int | None = None,
        hive_partitioning: bool | None = None,
        hive_schema: SchemaDict | None = None,
        try_parse_hive_dates: bool = True,
        include_file_paths: str | None = None,
        _record_batch_statistics: bool = False,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "scan_ipc",
                pl,
                source,
                n_rows=n_rows,
                cache=cache,
                rechunk=rechunk,
                row_index_name=row_index_name,
                row_index_offset=row_index_offset,
                glob=glob,
                storage_options=storage_options,
                credential_provider=credential_provider,
                memory_map=memory_map,
                retries=retries,
                file_cache_ttl=file_cache_ttl,
                hive_partitioning=hive_partitioning,
                hive_schema=hive_schema,
                try_parse_hive_dates=try_parse_hive_dates,
                include_file_paths=include_file_paths,
                _record_batch_statistics=_record_batch_statistics,
            )
        )

    def scan_lines(
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
        name: str = "lines",
        n_rows: int | None = None,
        row_index_name: str | None = None,
        row_index_offset: int = 0,
        glob: bool = True,
        storage_options: StorageOptionsDict | None = None,
        credential_provider: CredentialProviderFunction
        | Literal["auto"]
        | None = "auto",
        include_file_paths: str | None = None,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "scan_lines",
                pl,
                source,
                name=name,
                n_rows=n_rows,
                row_index_name=row_index_name,
                row_index_offset=row_index_offset,
                glob=glob,
                storage_options=storage_options,
                credential_provider=credential_provider,
                include_file_paths=include_file_paths,
            )
        )

    def scan_ndjson(
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
        storage_options: StorageOptionsDict | None = None,
        credential_provider: CredentialProviderFunction
        | Literal["auto"]
        | None = "auto",
        retries: int | None = None,
        file_cache_ttl: int | None = None,
        include_file_paths: str | None = None,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "scan_ndjson",
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

    def scan_parquet(
        self,
        source: FileSource,
        n_rows: int | None = None,
        row_index_name: str | None = None,
        row_index_offset: int = 0,
        parallel: ParallelStrategy = "auto",
        use_statistics: bool = True,
        hive_partitioning: bool | None = None,
        glob: bool = True,
        hidden_file_prefix: str | Sequence[str] | None = None,
        schema: SchemaDict | None = None,
        hive_schema: SchemaDict | None = None,
        try_parse_hive_dates: bool = True,
        rechunk: bool = False,
        low_memory: bool = False,
        cache: bool = True,
        storage_options: StorageOptionsDict | None = None,
        credential_provider: CredentialProviderFunction
        | Literal["auto"]
        | None = "auto",
        retries: int | None = None,
        include_file_paths: str | None = None,
        missing_columns: Literal["insert", "raise"] = "raise",
        allow_missing_columns: bool | None = None,
        extra_columns: Literal["ignore", "raise"] = "raise",
        cast_options: ScanCastOptions | None = None,
        _column_mapping: ColumnMapping | None = None,
        _default_values: DefaultFieldValues | None = None,
        _deletion_files: DeletionFiles | None = None,
        _table_statistics: DataFrame | None = None,
        _row_count: tuple[int, int] | None = None,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "scan_parquet",
                pl,
                source,
                n_rows=n_rows,
                row_index_name=row_index_name,
                row_index_offset=row_index_offset,
                parallel=parallel,
                use_statistics=use_statistics,
                hive_partitioning=hive_partitioning,
                glob=glob,
                hidden_file_prefix=hidden_file_prefix,
                schema=schema,
                hive_schema=hive_schema,
                try_parse_hive_dates=try_parse_hive_dates,
                rechunk=rechunk,
                low_memory=low_memory,
                cache=cache,
                storage_options=storage_options,
                credential_provider=credential_provider,
                retries=retries,
                include_file_paths=include_file_paths,
                missing_columns=missing_columns,
                allow_missing_columns=allow_missing_columns,
                extra_columns=extra_columns,
                cast_options=cast_options,
                _column_mapping=_column_mapping,
                _default_values=_default_values,
                _deletion_files=_deletion_files,
                _table_statistics=_table_statistics,
                _row_count=_row_count,
            )
        )

    # --- END INSERTION MARKER IN LazyPipeline
