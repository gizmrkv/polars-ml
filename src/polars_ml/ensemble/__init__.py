from typing import TYPE_CHECKING, Callable, Iterable

import polars as pl
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.model_selection import KFold
from polars_ml.pipeline.component import PipelineComponent

from .stacking import Stacking

if TYPE_CHECKING:
    from polars_ml import Pipeline

__all__ = ["Stacking"]


class EnsembleNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def stack(
        self,
        model_fn: Callable[[DataFrame, int], PipelineComponent],
        k_fold: KFold,
        *,
        aggs_on_transform: IntoExpr | Iterable[IntoExpr] = pl.all().mean(),
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            Stacking(
                model_fn,
                k_fold,
                aggs_on_transform=aggs_on_transform,
            ),
            component_name=component_name,
        )
