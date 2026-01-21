from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Self, Sequence, overload

from polars import DataFrame
from polars._typing import ColumnNameOrSelector, ConcatMethod, IntoExpr, JoinStrategy

from polars_ml.base import HasFeatureImportance, Transformer
from polars_ml.feature_engineering import FeatureEngineeringNameSpace
from polars_ml.gbdt import GBDTNameSpace
from polars_ml.metrics import MetricsNameSpace
from polars_ml.optimize import OptimizeNameSpace

from .basic import Apply, Const, Echo, Replay, Side
from .combine import Combine
from .concat import Concat
from .discretize import Discretize
from .horizontal import HorizontalNameSpace
from .join_agg import JoinAgg
from .label_encode import LabelEncode, LabelEncodeInverseContext
from .power import BoxCoxTransform, PowerTransformInverseContext, YeoJohnsonTransform
from .scale import MinMaxScale, RobustScale, ScaleInverseContext, StandardScale


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
