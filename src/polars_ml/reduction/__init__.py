from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

from polars import DataFrame
from polars._typing import IntoExpr

from .pca import PCA, PCAParameters
from .umap_ import UMAP, UMAPParameters

__all__ = ["PCA", "UMAP"]

if TYPE_CHECKING:
    from polars_ml import Pipeline


class ReductionNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def pca(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        *,
        prefix: str = "pca",
        include_input: bool = True,
        model_kwargs: PCAParameters
        | Callable[[DataFrame], PCAParameters]
        | None = None,
        out_dir: str | Path | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            PCA(
                features,
                prefix=prefix,
                include_input=include_input,
                model_kwargs=model_kwargs,
                out_dir=out_dir,
            ),
            component_name=component_name,
        )

    def umap(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        *,
        prefix: str = "umap",
        include_input: bool = True,
        model_kwargs: UMAPParameters
        | Callable[[DataFrame], UMAPParameters]
        | None = None,
        out_dir: str | Path | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            UMAP(
                features,
                prefix=prefix,
                include_input=include_input,
                model_kwargs=model_kwargs,
                out_dir=out_dir,
            ),
            component_name=component_name,
        )
