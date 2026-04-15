from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Self, Sequence, overload

from polars._typing import ColumnNameOrSelector, IntoExpr, JoinStrategy

from polars_ml import LazyTransformer

from .basic import Echo, Replay
from .combine import Combine
from .discretize import Discretize
from .horizontal import HorizontalAgg, HorizontalArgMax, HorizontalArgMin
from .join_agg import JoinAgg
from .label_encode import LabelEncode
from .power import BoxCoxTransform, YeoJohnsonTransform
from .scale import MinMaxScale, RobustScale, StandardScale


class PipelineMixin(ABC):
    @abstractmethod
    def pipe(self, step: LazyTransformer) -> Self: ...

    def echo(self) -> Self:
        return self.pipe(Echo())

    def replay(self) -> Self:
        return self.pipe(Replay())

    def min_max_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
    ) -> Self:
        return self.pipe(MinMaxScale(columns, *more_columns, by=by))

    def standard_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
    ) -> Self:
        return self.pipe(StandardScale(columns, *more_columns, by=by))

    def robust_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        quantile_range: tuple[float, float] = (0.25, 0.75),
    ) -> Self:
        return self.pipe(
            RobustScale(columns, *more_columns, by=by, quantile_range=quantile_range)
        )

    def boxcox(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
    ) -> Self:
        return self.pipe(BoxCoxTransform(columns, *more_columns, by=by))

    def yeojohnson(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
    ) -> Self:
        return self.pipe(YeoJohnsonTransform(columns, *more_columns, by=by))

    def label_encode(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        orders: Mapping[str, Sequence[Any]] | None = None,
        maintain_order: bool = True,
    ) -> Self:
        return self.pipe(
            LabelEncode(
                columns, *more_columns, orders=orders, maintain_order=maintain_order
            )
        )

    def horizontal_agg(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_agg",
        variable_name: str | None = None,
        maintain_order: bool = True,
        aggs: Iterable[IntoExpr | Iterable[IntoExpr]] | None = None,
        named_aggs: Mapping[str, IntoExpr] | None = None,
    ) -> Self:
        return self.pipe(
            HorizontalAgg(
                expr,
                *more_expr,
                value_name=value_name,
                variable_name=variable_name,
                maintain_order=maintain_order,
                aggs=aggs,
                named_aggs=named_aggs,
            )
        )

    def horizontal_argmax(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_argmax",
        maintain_order: bool = True,
    ) -> Self:
        return self.pipe(
            HorizontalArgMax(
                expr, *more_expr, value_name=value_name, maintain_order=maintain_order
            )
        )

    def horizontal_argmin(
        self,
        expr: IntoExpr | Iterable[IntoExpr],
        *more_expr: IntoExpr | Iterable[IntoExpr],
        value_name: str = "horizontal_argmin",
        maintain_order: bool = True,
    ) -> Self:
        return self.pipe(
            HorizontalArgMin(
                expr, *more_expr, value_name=value_name, maintain_order=maintain_order
            )
        )

    def combine(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        n: int,
        *,
        delimiter: str = "_",
    ) -> Self:
        return self.pipe(Combine(columns, n, delimiter=delimiter))

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
