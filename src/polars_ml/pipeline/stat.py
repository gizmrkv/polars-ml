from abc import ABC
from typing import TYPE_CHECKING, Generic, Iterable, Sequence

import polars as pl
from polars._typing import IntoExpr, RollingInterpolationMethod

from polars_ml.typing import PipelineType
from polars_ml.utils import LazyGetAttrWithName

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
        include_name: bool = False,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyGetAttrWithName(
                "bottom_k", k, by=by, reverse=reverse, include_name=include_name
            )
        )

    def top_k(
        self,
        k: int,
        *,
        by: IntoExpr | Iterable[IntoExpr],
        reverse: bool | Sequence[bool] = False,
        include_name: bool = False,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyGetAttrWithName(
                "top_k", k, by=by, reverse=reverse, include_name=include_name
            )
        )

    def count(self, include_name: bool = False) -> PipelineType:
        return self.pipeline.pipe(
            LazyGetAttrWithName("count", include_name=include_name)
        )

    def null_count(self, include_name: bool = False) -> PipelineType:
        return self.pipeline.pipe(
            LazyGetAttrWithName("null_count", include_name=include_name)
        )

    def n_unique(self, include_name: bool = False) -> PipelineType:
        return self.pipeline.pipe(
            LazyGetAttrWithName(
                "select", pl.all().n_unique(), include_name=include_name
            )
        )

    def quantile(
        self,
        quantile: float,
        interpolation: RollingInterpolationMethod = "nearest",
        include_name: bool = False,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyGetAttrWithName(
                "quantile",
                quantile,
                interpolation=interpolation,
                include_name=include_name,
            )
        )

    def max(self, include_name: bool = False) -> PipelineType:
        return self.pipeline.pipe(LazyGetAttrWithName("max", include_name=include_name))

    def min(self, include_name: bool = False) -> PipelineType:
        return self.pipeline.pipe(LazyGetAttrWithName("min", include_name=include_name))

    def median(self, include_name: bool = False) -> PipelineType:
        return self.pipeline.pipe(
            LazyGetAttrWithName("median", include_name=include_name)
        )

    def sum(self, include_name: bool = False) -> PipelineType:
        return self.pipeline.pipe(LazyGetAttrWithName("sum", include_name=include_name))

    def mean(self, include_name: bool = False) -> PipelineType:
        return self.pipeline.pipe(
            LazyGetAttrWithName("mean", include_name=include_name)
        )

    def std(self, ddof: int = 1, include_name: bool = False) -> PipelineType:
        return self.pipeline.pipe(
            LazyGetAttrWithName("std", ddof=ddof, include_name=include_name)
        )

    def var(self, ddof: int = 1, include_name: bool = False) -> PipelineType:
        return self.pipeline.pipe(
            LazyGetAttrWithName("var", ddof=ddof, include_name=include_name)
        )


class StatNameSpace(BaseStatNameSpace["Pipeline"]):
    pass


class LazyStatNameSpace(BaseStatNameSpace["LazyPipeline"]):
    pass
