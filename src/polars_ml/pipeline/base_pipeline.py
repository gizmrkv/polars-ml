from abc import ABC
from datetime import timedelta
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    Literal,
    Mapping,
    Self,
    Sequence,
    override,
)

import numpy as np
from polars import DataFrame, Expr, LazyFrame
from polars._typing import (
    AsofJoinStrategy,
    ClosedInterval,
    ColumnNameOrSelector,
    FillNullStrategy,
    IntoExpr,
    IntoExprColumn,
    JoinStrategy,
    JoinValidation,
    Label,
    PolarsDataType,
    StartBy,
    UniqueKeepStrategy,
)

from polars_ml.component import BaseComponent, ComponentList, LazyComponent
from polars_ml.utils import LazyGetAttr

from .group_by import LazyGroupByNameSpace


class BasePipeline(BaseComponent, ABC):
    def __init__(self):
        self.components = ComponentList[LazyComponent]()

    @override
    def is_fitted(self) -> bool:
        return self.components.is_fitted()

    @override
    def set_log_dir(self, log_dir: str | Path | None) -> Self:
        super().set_log_dir(log_dir)
        self.components.set_log_dir(log_dir)
        return self

    def save(
        self,
        log_dir: Path,
        filename: str,
        *,
        extension: Literal[".z", ".gz", ".bz2", ".xz", ".lzma"] = ".z",
        compress: int = 0,
        protocol: int | None = None,
    ):
        import joblib

        log_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            self,
            log_dir / (filename + extension),
            compress=compress,
            protocol=protocol,
        )

    def pipe(self, *components: LazyComponent) -> Self:
        self.components.extend(components)
        return self

    def cast(
        self,
        dtypes: (
            Mapping[ColumnNameOrSelector | PolarsDataType, PolarsDataType]
            | PolarsDataType
        ),
        *,
        strict: bool = True,
    ) -> Self:
        return self.pipe(
            LazyGetAttr("cast", dtypes, strict=strict).set_component_name("Cast")
        )

    def clear(self, n: int = 0) -> Self:
        return self.pipe(LazyGetAttr("clear", n).set_component_name("Clear"))

    def clone(self) -> Self:
        return self.pipe(LazyGetAttr("clone").set_component_name("Clone"))

    def drop(
        self,
        *columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        strict: bool = True,
    ) -> Self:
        return self.pipe(
            LazyGetAttr("drop", columns, strict=strict).set_component_name("Drop")
        )

    def drop_nulls(
        self,
        subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
    ) -> Self:
        return self.pipe(
            LazyGetAttr("drop_nulls", subset).set_component_name("DropNulls")
        )

    def explode(
        self,
        columns: str | Expr | Sequence[str | Expr],
        *more_columns: str | Expr,
    ) -> Self:
        return self.pipe(
            LazyGetAttr("explode", columns, *more_columns).set_component_name("Explode")
        )

    def fill_nan(self, value: Expr | int | float | None) -> Self:
        return self.pipe(LazyGetAttr("fill_nan", value).set_component_name("FillNan"))

    def fill_null(
        self,
        value: Any | Expr | None = None,
        strategy: FillNullStrategy | None = None,
        limit: int | None = None,
        *,
        matches_supertype: bool = True,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "fill_null", value, strategy, limit, matches_supertype=matches_supertype
            ).set_component_name("FillNull")
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
        return self.pipe(
            LazyGetAttr("filter", *predicates, **constraints).set_component_name(
                "Filter"
            )
        )

    def gather_every(self, n: int, offset: int = 0) -> Self:
        return self.pipe(
            LazyGetAttr("gather_every", n, offset).set_component_name("GatherEvery")
        )

    def slice(self, offset: int, length: int | None = None) -> Self:
        return self.pipe(
            LazyGetAttr("slice", offset, length).set_component_name("Slice")
        )

    def head(self, n: int = 5) -> Self:
        return self.pipe(LazyGetAttr("head", n).set_component_name("Head"))

    def limit(self, n: int = 5) -> Self:
        return self.pipe(LazyGetAttr("limit", n).set_component_name("Limit"))

    def tail(self, n: int = 5) -> Self:
        return self.pipe(LazyGetAttr("tail", n).set_component_name("Tail"))

    def interpolate(self) -> Self:
        return self.pipe(LazyGetAttr("interpolate").set_component_name("Interpolate"))

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
            LazyGetAttr(
                "sort",
                by,
                *more_by,
                descending=descending,
                nulls_last=nulls_last,
                multithreaded=multithreaded,
                maintain_order=maintain_order,
            ).set_component_name("Sort")
        )

    def set_sorted(self, column: str, *, descending: bool = False) -> Self:
        return self.pipe(
            LazyGetAttr("set_sorted", column, descending=descending).set_component_name(
                "SetSorted"
            )
        )

    def rename(self, mapping: dict[str, str] | Callable[[str], str]) -> Self:
        return self.pipe(LazyGetAttr("rename", mapping).set_component_name("Rename"))

    def reverse(self) -> Self:
        return self.pipe(LazyGetAttr("reverse").set_component_name("Reverse"))

    def select(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.pipe(
            LazyGetAttr("select", *exprs, **named_exprs).set_component_name("Select")
        )

    def select_seq(
        self, *exprs: IntoExpr | Iterable[IntoExpr], **named_exprs: IntoExpr
    ) -> Self:
        return self.pipe(
            LazyGetAttr("select_seq", *exprs, **named_exprs).set_component_name(
                "SelectSeq"
            )
        )

    def shift(self, n: int = 1, *, fill_value: IntoExpr | None = None) -> Self:
        return self.pipe(
            LazyGetAttr("shift", n, fill_value=fill_value).set_component_name("Shift")
        )

    def sql(self, query: str, *, table_name: str = "self") -> Self:
        return self.pipe(
            LazyGetAttr("sql", query, table_name=table_name).set_component_name("SQL")
        )

    def unique(
        self,
        subset: ColumnNameOrSelector | Collection[ColumnNameOrSelector] | None = None,
        *,
        keep: UniqueKeepStrategy = "any",
        maintain_order: bool = False,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "unique", subset, keep=keep, maintain_order=maintain_order
            ).set_component_name("Unique")
        )

    def unnest(
        self,
        columns: ColumnNameOrSelector | Collection[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector,
    ) -> Self:
        return self.pipe(
            LazyGetAttr("unnest", columns, *more_columns).set_component_name("Unnest")
        )

    def unpivot(
        self,
        on: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        *,
        index: ColumnNameOrSelector | Sequence[ColumnNameOrSelector] | None = None,
        variable_name: str | None = None,
        value_name: str | None = None,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "unpivot",
                on,
                index=index,
                variable_name=variable_name,
                value_name=value_name,
            ).set_component_name("Unpivot")
        )

    def with_columns(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> Self:
        return self.pipe(
            LazyGetAttr("with_columns", *exprs, **named_exprs).set_component_name(
                "WithColumns"
            )
        )

    def with_columns_seq(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        **named_exprs: IntoExpr,
    ) -> Self:
        return self.pipe(
            LazyGetAttr("with_columns_seq", *exprs, **named_exprs).set_component_name(
                "WithColumnsSeq"
            )
        )

    def with_row_index(self, name: str = "index", offset: int = 0) -> Self:
        return self.pipe(
            LazyGetAttr("with_row_index", name, offset).set_component_name(
                "WithRowIndex"
            )
        )

    def group_by(
        self,
        *by: IntoExpr | Iterable[IntoExpr],
        maintain_order: bool = False,
        **named_by: IntoExpr,
    ) -> LazyGroupByNameSpace[Self]:
        return LazyGroupByNameSpace(
            self, "group_by", *by, **named_by, maintain_order=maintain_order
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
    ) -> LazyGroupByNameSpace[Self]:
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
    ) -> LazyGroupByNameSpace[Self]:
        return LazyGroupByNameSpace(
            self,
            "rolling",
            index_column,
            period=period,
            offset=offset,
            closed=closed,
            group_by=group_by,
        )

    def join(
        self,
        other: DataFrame | LazyFrame,
        on: str | Expr | Sequence[str | Expr] | None = None,
        how: JoinStrategy = "inner",
        *,
        left_on: str | Expr | Sequence[str | Expr] | None = None,
        right_on: str | Expr | Sequence[str | Expr] | None = None,
        suffix: str = "_right",
        validate: JoinValidation = "m:m",
        join_nulls: bool = False,
        coalesce: bool | None = None,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "join",
                other.lazy(),
                on,
                how,
                left_on=left_on,
                right_on=right_on,
                suffix=suffix,
                validate=validate,
                join_nulls=join_nulls,
                coalesce=coalesce,
            ).set_component_name("Join")
        )

    def join_asof(
        self,
        other: DataFrame | LazyFrame,
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
            LazyGetAttr(
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
            ).set_component_name("JoinAsof")
        )

    def merge_sorted(self, other: DataFrame | LazyFrame, key: str) -> Self:
        return self.pipe(
            LazyGetAttr("merge_sorted", other, key).set_component_name("MergeSorted")
        )

    def update(
        self,
        other: DataFrame | LazyFrame,
        on: str | Sequence[str] | None = None,
        how: Literal["left", "inner", "full"] = "left",
        *,
        left_on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        include_nulls: bool = False,
    ) -> Self:
        return self.pipe(
            LazyGetAttr(
                "update",
                other,
                on,
                how,
                left_on=left_on,
                right_on=right_on,
                include_nulls=include_nulls,
            ).set_component_name("Update")
        )
