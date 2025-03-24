from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import umap
from polars import DataFrame
from polars._typing import IntoExpr
from sklearn import decomposition

from .pca import PCA
from .umap_ import UMAP

__all__ = ["PCA", "UMAP"]

if TYPE_CHECKING:
    from polars_ml import Pipeline


class ReductionNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def pca(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        pca: decomposition.PCA,
        *,
        prefix: str = "pca",
        include_input: bool = True,
        out_dir: str | Path | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            PCA(
                features,
                pca,
                prefix=prefix,
                include_input=include_input,
                out_dir=out_dir,
            ),
            component_name=component_name,
        )

    def umap(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        umap: umap.UMAP,
        *,
        prefix: str = "umap",
        include_input: bool = True,
        out_dir: str | Path | None = None,
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            UMAP(
                features,
                umap,
                prefix=prefix,
                include_input=include_input,
                out_dir=out_dir,
            ),
            component_name=component_name,
        )
