from abc import ABC
from typing import TYPE_CHECKING, Generic, Iterable, Sequence

import polars as pl
from polars._typing import IntoExpr, RollingInterpolationMethod

from polars_ml.typing import PipelineType
from polars_ml.utils import LazyGetAttr

if TYPE_CHECKING:
    from .lazy_pipeline import LazyPipeline  # noqa: F401
    from .pipeline import Pipeline  # noqa: F401


class BaseStatNameSpace(Generic[PipelineType], ABC):
    def __init__(self, pipeline: PipelineType):
        self.pipeline = pipeline

    def bottom_k(
        self,
        k: int,
        *,
        by: IntoExpr | Iterable[IntoExpr],
        reverse: bool | Sequence[bool] = False,
    ) -> PipelineType:
        return self.pipeline.pipe(LazyGetAttr("bottom_k", k, by=by, reverse=reverse))

    def top_k(
        self,
        k: int,
        *,
        by: IntoExpr | Iterable[IntoExpr],
        reverse: bool | Sequence[bool] = False,
    ) -> PipelineType:
        return self.pipeline.pipe(LazyGetAttr("top_k", k, by=by, reverse=reverse))

    def count(self) -> PipelineType:
        return self.pipeline.pipe(LazyGetAttr("count"))

    def null_count(self) -> PipelineType:
        return self.pipeline.pipe(LazyGetAttr("null_count"))

    def n_unique(self) -> PipelineType:
        return self.pipeline.pipe(LazyGetAttr("select", pl.all().n_unique()))

    def quantile(
        self,
        quantile: float,
        interpolation: RollingInterpolationMethod = "nearest",
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyGetAttr("quantile", quantile, interpolation=interpolation)
        )

    def max(self) -> PipelineType:
        return self.pipeline.pipe(LazyGetAttr("max"))

    def min(self) -> PipelineType:
        return self.pipeline.pipe(LazyGetAttr("min"))

    def median(self) -> PipelineType:
        return self.pipeline.pipe(LazyGetAttr("median"))

    def sum(self) -> PipelineType:
        return self.pipeline.pipe(LazyGetAttr("sum"))

    def mean(self) -> PipelineType:
        return self.pipeline.pipe(LazyGetAttr("mean"))

    def std(self, ddof: int = 1) -> PipelineType:
        return self.pipeline.pipe(LazyGetAttr("std", ddof=ddof))

    def var(self, ddof: int = 1) -> PipelineType:
        return self.pipeline.pipe(LazyGetAttr("var", ddof=ddof))


class StatNameSpace(BaseStatNameSpace["Pipeline"]):
    pass


class LazyStatNameSpace(BaseStatNameSpace["LazyPipeline"]):
    pass
