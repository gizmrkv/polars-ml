from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Self, Sequence, overload

import polars as pl
from polars._typing import ColumnNameOrSelector

from polars_ml import LazyTransformer

from .basic import Echo, Replay
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
