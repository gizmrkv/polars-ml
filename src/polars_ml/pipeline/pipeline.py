from __future__ import annotations

from typing import Callable, Self, Sequence

from polars import DataFrame
from polars._typing import ConcatMethod

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
from .label_encode import LabelEncode, LabelEncodeInverse, LabelEncodeInverseContext
from .power import (
    BoxCoxTransform,
    PowerTransformInverse,
    PowerTransformInverseContext,
    YeoJohnsonTransform,
)
from .scale import (
    MinMaxScale,
    RobustScale,
    ScaleInverse,
    ScaleInverseContext,
    StandardScale,
)


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
