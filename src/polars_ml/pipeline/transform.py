from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, Generic, Iterable, Sequence, Tuple

from polars._typing import ColumnNameOrSelector, IntoExpr

from polars_ml.transform import (
    MoveScaler,
    OrdinalEncoder,
    TargetEncoder,
    Transformer,
    TransformerMixin,
)
from polars_ml.typing import PipelineType

if TYPE_CHECKING:
    from .lazy_pipeline import LazyPipeline  # noqa: F401
    from .pipeline import Pipeline  # noqa: F401


class BaseTransformNameSpace(Generic[PipelineType], ABC):
    def __init__(self, pipeline: PipelineType):
        self.pipeline = pipeline

    def label_encode(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        maintain_order: bool = False,
    ) -> PipelineType:
        return self.pipeline.pipe(OrdinalEncoder(*exprs, maintain_order=maintain_order))

    def ordinal_encode(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        orders: Dict[str, Sequence[Any]] | None = None,
    ) -> PipelineType:
        return self.pipeline.pipe(OrdinalEncoder(*exprs, orders=orders))

    def target_encode(
        self,
        by: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> PipelineType:
        return self.pipeline.pipe(TargetEncoder(by, *aggs, **named_aggs))

    def min_max_scale(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> PipelineType:
        return self.pipeline.pipe(MoveScaler(*exprs, method="min_max"))

    def standard_scale(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> PipelineType:
        return self.pipeline.pipe(MoveScaler(*exprs, method="standard"))

    def robust_scale(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        quantile: Tuple[float, float] = (0.25, 0.75),
    ) -> PipelineType:
        return self.pipeline.pipe(
            MoveScaler(*exprs, method="robust", quantile=quantile)
        )


class TransformNameSpace(BaseTransformNameSpace["Pipeline"]):
    def pipe(
        self,
        transformer: TransformerMixin,
        *,
        X: IntoExpr | Iterable[IntoExpr],
        y: IntoExpr | None = None,
        name: str = "out",
    ) -> "Pipeline":
        return self.pipeline.pipe(Transformer(transformer, X=X, y=y, name=name))


class LazyTransformNameSpace(BaseTransformNameSpace["LazyPipeline"]):
    pass
