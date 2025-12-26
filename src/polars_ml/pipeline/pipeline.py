from __future__ import annotations

from datetime import datetime, timedelta
from io import IOBase
from pathlib import Path
from typing import (
    IO,
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
    AvroCompression,
    ClosedInterval,
    ColumnNameOrSelector,
    ConcatMethod,
    ConnectionOrCursor,
    CorrelationMethod,
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
    ParallelStrategy,
    ParquetCompression,
    ParquetMetadata,
    PivotAgg,
    PolarsDataType,
    PythonDataType,
    QuantileMethod,
    SchemaDefinition,
    SchemaDict,
    SizeUnit,
    StartBy,
    UniqueKeepStrategy,
    UnstackDirection,
)
from polars.interchange.protocol import CompatLevel
from polars.io.cloud import CredentialProviderFunction

from polars_ml.base import Transformer
from polars_ml.gbdt import GBDTNameSpace
from polars_ml.metrics import MetricsNameSpace
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
from .getattr import GetAttr, GetAttrPolars
from .group_by import DynamicGroupByNameSpace, GroupByNameSpace, RollingGroupByNameSpace


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

    @property
    def metrics(self) -> MetricsNameSpace:
        return MetricsNameSpace(self)

    # --- START INSERTION MARKER IN Pipeline

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

    def write_avro(
        self,
        file: str | Path | IO[bytes],
        compression: AvroCompression = "uncompressed",
        name: str = "",
    ) -> Self:
        return self.pipe(GetAttr("write_avro", file, compression, name))

    def write_clipboard(self, separator: str = "\t", **kwargs: Any) -> Self:
        return self.pipe(GetAttr("write_clipboard", separator=separator, **kwargs))

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
                table_name,
                connection,
                if_table_exists=if_table_exists,
                engine=engine,
                engine_options=engine_options,
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
                file,
                compression=compression,
                compat_level=compat_level,
            )
        )

    def write_json(self, file: IOBase | str | Path | None = None) -> Self:
        return self.pipe(GetAttr("write_json", file))

    def write_ndjson(
        self, file: str | Path | IO[bytes] | IO[str] | None = None
    ) -> Self:
        return self.pipe(GetAttr("write_ndjson", file))

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
            GetAttrPolars("read_avro", source, columns=columns, n_rows=n_rows)
        )

    def read_clipboard(self, separator: str = "\t", **kwargs: Any) -> Self:
        return self.pipe(GetAttrPolars("read_clipboard", separator, **kwargs))

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
            GetAttrPolars(
                "read_csv",
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

    def read_csv_batched(
        self,
        source: str | Path,
        has_header: bool = True,
        columns: Sequence[int] | Sequence[str] | None = None,
        new_columns: Sequence[str] | None = None,
        separator: str = ",",
        comment_prefix: str | None = None,
        quote_char: str | None = '"',
        skip_rows: int = 0,
        skip_lines: int = 0,
        schema_overrides: Mapping[str, PolarsDataType]
        | Sequence[PolarsDataType]
        | None = None,
        null_values: str | Sequence[str] | dict[str, str] | None = None,
        missing_utf8_is_empty_string: bool = False,
        ignore_errors: bool = False,
        try_parse_dates: bool = False,
        n_threads: int | None = None,
        infer_schema_length: int | None = 100,
        batch_size: int = 50000,
        n_rows: int | None = None,
        encoding: CsvEncoding | str = "utf8",
        low_memory: bool = False,
        rechunk: bool = False,
        skip_rows_after_header: int = 0,
        row_index_name: str | None = None,
        row_index_offset: int = 0,
        sample_size: int = 1024,
        eol_char: str = "\n",
        raise_if_empty: bool = True,
        truncate_ragged_lines: bool = False,
        decimal_comma: bool = False,
    ) -> Self:
        return self.pipe(
            GetAttrPolars(
                "read_csv_batched",
                source,
                has_header=has_header,
                columns=columns,
                new_columns=new_columns,
                separator=separator,
                comment_prefix=comment_prefix,
                quote_char=quote_char,
                skip_rows=skip_rows,
                skip_lines=skip_lines,
                schema_overrides=schema_overrides,
                null_values=null_values,
                missing_utf8_is_empty_string=missing_utf8_is_empty_string,
                ignore_errors=ignore_errors,
                try_parse_dates=try_parse_dates,
                n_threads=n_threads,
                infer_schema_length=infer_schema_length,
                batch_size=batch_size,
                n_rows=n_rows,
                encoding=encoding,
                low_memory=low_memory,
                rechunk=rechunk,
                skip_rows_after_header=skip_rows_after_header,
                row_index_name=row_index_name,
                row_index_offset=row_index_offset,
                sample_size=sample_size,
                eol_char=eol_char,
                raise_if_empty=raise_if_empty,
                truncate_ragged_lines=truncate_ragged_lines,
                decimal_comma=decimal_comma,
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
            GetAttrPolars(
                "read_database_uri",
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
            GetAttrPolars(
                "read_ipc",
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

    def read_ipc_schema(self, source: str | Path | IO[bytes] | bytes) -> Self:
        return self.pipe(GetAttrPolars("read_ipc_schema", source))

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
            GetAttrPolars(
                "read_ipc_stream",
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
            GetAttrPolars(
                "read_json",
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
            GetAttrPolars(
                "read_ndjson",
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

    def read_ods(
        self,
        source: FileSource,
        sheet_id: int | Sequence[int] | None = None,
        sheet_name: str | list[str] | tuple[str] | None = None,
        has_header: bool = True,
        columns: Sequence[int] | Sequence[str] | None = None,
        schema_overrides: SchemaDict | None = None,
        infer_schema_length: int | None = 100,
        include_file_paths: str | None = None,
        drop_empty_rows: bool = True,
        drop_empty_cols: bool = True,
        raise_if_empty: bool = True,
    ) -> Self:
        return self.pipe(
            GetAttrPolars(
                "read_ods",
                source,
                sheet_id=sheet_id,
                sheet_name=sheet_name,
                has_header=has_header,
                columns=columns,
                schema_overrides=schema_overrides,
                infer_schema_length=infer_schema_length,
                include_file_paths=include_file_paths,
                drop_empty_rows=drop_empty_rows,
                drop_empty_cols=drop_empty_cols,
                raise_if_empty=raise_if_empty,
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
            GetAttrPolars(
                "read_parquet",
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

    def read_parquet_metadata(
        self,
        source: str | Path | IO[bytes] | bytes,
        storage_options: dict[str, Any] | None = None,
        credential_provider: CredentialProviderFunction
        | Literal["auto"]
        | None = "auto",
        retries: int = 2,
    ) -> Self:
        return self.pipe(
            GetAttrPolars(
                "read_parquet_metadata",
                source,
                storage_options,
                credential_provider,
                retries,
            )
        )

    def read_parquet_schema(self, source: str | Path | IO[bytes] | bytes) -> Self:
        return self.pipe(GetAttrPolars("read_parquet_schema", source))

    def group_by(
        self,
        *by: IntoExpr | Iterable[IntoExpr],
        maintain_order: bool = False,
        **named_by: IntoExpr,
    ) -> GroupByNameSpace:
        return GroupByNameSpace(
            self, "group_by", *by, maintain_order=maintain_order, **named_by
        )

    def group_by_dynamic(
        self,
        index_column: IntoExpr,
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
        period: str | timedelta,
        offset: str | timedelta | None = None,
        closed: ClosedInterval = "right",
        group_by: IntoExpr | Iterable[IntoExpr] | None = None,
    ) -> RollingGroupByNameSpace:
        return RollingGroupByNameSpace(
            self,
            "rolling",
            index_column,
            period=period,
            offset=offset,
            closed=closed,
            group_by=group_by,
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

    # --- END INSERTION MARKER IN Pipeline
