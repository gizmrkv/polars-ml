import uuid
from abc import ABC
from typing import TYPE_CHECKING, Dict, Generic, Iterable

import polars as pl
from polars import Expr
from polars._typing import IntoExpr, RollingInterpolationMethod

from polars_ml.typing import PipelineType
from polars_ml.utils import LazyGetAttr, LazyHorizontalAgg

if TYPE_CHECKING:
    from .lazy_pipeline import LazyPipeline  # noqa: F401
    from .pipeline import Pipeline  # noqa: F401


class BaseHorizontalNameSpace(Generic[PipelineType], ABC):
    def __init__(self, pipeline: PipelineType):
        self.pipeline = pipeline

    def agg(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        variable_name: str | None = None,
        value_name: str | None = None,
        maintain_order: bool = True,
        aggs: Iterable[IntoExpr | Iterable[IntoExpr]] | None = None,
        named_aggs: Dict[str, IntoExpr] | None = None,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                variable_name=variable_name,
                value_name=value_name,
                maintain_order=maintain_order,
                aggs=aggs,
                named_aggs=named_aggs,
            )
        )

    def all(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        value_name: str = "all",
        ignore_nulls: bool = True,
        maintain_order: bool = True,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                value_name=value_name,
                maintain_order=maintain_order,
                aggs=[pl.col(value_name).all(ignore_nulls=ignore_nulls)],
            )
        )

    def any(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        value_name: str = "any",
        ignore_nulls: bool = True,
        maintain_order: bool = True,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                value_name=value_name,
                maintain_order=maintain_order,
                aggs=[pl.col(value_name).any(ignore_nulls=ignore_nulls)],
            )
        )

    def count(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        value_name: str = "count",
        maintain_order: bool = True,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                value_name=value_name,
                maintain_order=maintain_order,
                aggs=[pl.col(value_name).count()],
            )
        )

    def null_count(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        value_name: str = "null_count",
        maintain_order: bool = True,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                value_name=value_name,
                maintain_order=maintain_order,
                aggs=[pl.col(value_name).null_count()],
            )
        )

    def n_unique(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        value_name: str = "n_unique",
        maintain_order: bool = True,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                value_name=value_name,
                maintain_order=maintain_order,
                aggs=[pl.col(value_name).n_unique()],
            )
        )

    def max(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        value_name: str = "max",
        maintain_order: bool = True,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                value_name=value_name,
                maintain_order=maintain_order,
                aggs=[pl.col(value_name).max()],
            )
        )

    def min(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        value_name: str = "min",
        maintain_order: bool = True,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                value_name=value_name,
                maintain_order=maintain_order,
                aggs=[pl.col(value_name).min()],
            )
        )

    def nan_max(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        value_name: str = "nan_max",
        maintain_order: bool = True,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                value_name=value_name,
                maintain_order=maintain_order,
                aggs=[pl.col(value_name).nan_max()],
            )
        )

    def nan_min(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        value_name: str = "nan_min",
        maintain_order: bool = True,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                value_name=value_name,
                maintain_order=maintain_order,
                aggs=[pl.col(value_name).nan_min()],
            )
        )

    def arg_max(
        self, *exprs: IntoExpr | Iterable[IntoExpr], value_name: str = "arg_max"
    ) -> PipelineType:
        variable_name = uuid.uuid4().hex
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                value_name=value_name,
                variable_name=variable_name,
                aggs=[
                    pl.struct(value_name, variable_name).filter(
                        pl.col(value_name) == pl.col(value_name).max()
                    )
                ],
            )
        ).pipe(
            LazyGetAttr(
                "with_columns",
                pl.col(value_name).list.eval(pl.element().struct.field(variable_name)),
            )
        )

    def arg_min(
        self, *exprs: IntoExpr | Iterable[IntoExpr], value_name: str = "arg_min"
    ) -> PipelineType:
        variable_name = uuid.uuid4().hex
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                value_name=value_name,
                variable_name=variable_name,
                aggs=[
                    pl.struct(value_name, variable_name).filter(
                        pl.col(value_name) == pl.col(value_name).min()
                    )
                ],
            )
        ).pipe(
            LazyGetAttr(
                "with_columns",
                pl.col(value_name).list.eval(pl.element().struct.field(variable_name)),
            )
        )

    def median(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        value_name: str = "median",
        maintain_order: bool = True,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                value_name=value_name,
                maintain_order=maintain_order,
                aggs=[pl.col(value_name).median()],
            )
        )

    def quantile(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        value_name: str = "quantile",
        maintain_order: bool = True,
        quantile: float | Expr,
        interpolation: RollingInterpolationMethod = "nearest",
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                value_name=value_name,
                maintain_order=maintain_order,
                aggs=[pl.col(value_name).quantile(quantile, interpolation)],
            )
        )

    def mean(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        value_name: str = "mean",
        maintain_order: bool = True,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                value_name=value_name,
                maintain_order=maintain_order,
                aggs=[pl.col(value_name).mean()],
            )
        )

    def sum(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        value_name: str = "sum",
        maintain_order: bool = True,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                value_name=value_name,
                maintain_order=maintain_order,
                aggs=[pl.col(value_name).sum()],
            )
        )

    def std(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        value_name: str = "std",
        maintain_order: bool = True,
        ddof: int = 1,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                value_name=value_name,
                maintain_order=maintain_order,
                aggs=[pl.col(value_name).std(ddof=ddof)],
            )
        )

    def var(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        value_name: str = "var",
        maintain_order: bool = True,
        ddof: int = 1,
    ) -> PipelineType:
        return self.pipeline.pipe(
            LazyHorizontalAgg(
                *exprs,
                value_name=value_name,
                maintain_order=maintain_order,
                aggs=[pl.col(value_name).var(ddof=ddof)],
            )
        )


class HorizontalNameSpace(BaseHorizontalNameSpace["Pipeline"]):
    pass


class LazyHorizontalNameSpace(BaseHorizontalNameSpace["LazyPipeline"]):
    pass
