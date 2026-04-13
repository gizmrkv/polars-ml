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
from .label_encode import LabelEncode, LabelEncodeInverseContext
from .power import BoxCoxTransform, PowerTransformInverseContext, YeoJohnsonTransform
from .scale import MinMaxScale, RobustScale, ScaleInverseContext, StandardScale


class PipelineMixin(ABC):
    @abstractmethod
    def pipe(self, step: LazyTransformer) -> Self: ...

    def echo(self) -> Self:
        return self.pipe(Echo())

    def replay(self) -> Self:
        return self.pipe(Replay())

    @overload
    def min_max_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
    ) -> Self: ...

    @overload
    def min_max_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        inverse_mapping: Mapping[str, str] | None,
    ) -> ScaleInverseContext: ...

    def min_max_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        inverse_mapping: Mapping[str, str] | None = None,
    ) -> Self | ScaleInverseContext:
        step = MinMaxScale(columns, *more_columns, by=by)
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
    ) -> Self: ...

    @overload
    def standard_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        inverse_mapping: Mapping[str, str] | None,
    ) -> ScaleInverseContext: ...

    def standard_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        inverse_mapping: Mapping[str, str] | None = None,
    ) -> Self | ScaleInverseContext:
        step = StandardScale(columns, *more_columns, by=by)
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
    ) -> Self: ...

    @overload
    def robust_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        quantile_range: tuple[float, float] = (0.25, 0.75),
        inverse_mapping: Mapping[str, str] | None,
    ) -> ScaleInverseContext: ...

    def robust_scale(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        quantile_range: tuple[float, float] = (0.25, 0.75),
        inverse_mapping: Mapping[str, str] | None = None,
    ) -> Self | ScaleInverseContext:
        step = RobustScale(columns, *more_columns, by=by, quantile_range=quantile_range)
        if inverse_mapping is None:
            return self.pipe(step)
        else:
            return ScaleInverseContext(self, step, inverse_mapping)

    @overload
    def boxcox(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
    ) -> Self: ...

    @overload
    def boxcox(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        inverse_mapping: Mapping[str, str] | None,
    ) -> PowerTransformInverseContext: ...

    def boxcox(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        inverse_mapping: Mapping[str, str] | None = None,
    ) -> Self | PowerTransformInverseContext:
        step = BoxCoxTransform(columns, *more_columns, by=by)
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
    ) -> Self: ...

    @overload
    def yeojohnson(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        inverse_mapping: Mapping[str, str] | None,
    ) -> PowerTransformInverseContext: ...

    def yeojohnson(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        inverse_mapping: Mapping[str, str] | None = None,
    ) -> Self | PowerTransformInverseContext:
        step = YeoJohnsonTransform(columns, *more_columns, by=by)
        if inverse_mapping is None:
            return self.pipe(step)
        else:
            return PowerTransformInverseContext(self, step, inverse_mapping)

    @overload
    def label_encode(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        orders: Mapping[str, Sequence[Any]] | None = None,
        maintain_order: bool = True,
    ) -> Self: ...

    @overload
    def label_encode(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        orders: Mapping[str, Sequence[Any]] | None = None,
        maintain_order: bool = True,
        inverse_mapping: Mapping[str, str] | None,
    ) -> LabelEncodeInverseContext: ...

    def label_encode(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        orders: Mapping[str, Sequence[Any]] | None = None,
        maintain_order: bool = True,
        inverse_mapping: Mapping[str, str] | None = None,
    ) -> Self | LabelEncodeInverseContext:
        step = LabelEncode(
            columns,
            *more_columns,
            orders=orders,
            maintain_order=maintain_order,
        )
        if inverse_mapping is None:
            return self.pipe(step)
        else:
            return LabelEncodeInverseContext(self, step, inverse_mapping)

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
