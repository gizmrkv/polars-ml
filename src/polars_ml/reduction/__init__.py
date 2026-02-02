from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Sequence

from polars._typing import ColumnNameOrSelector

from .pca import PCA
from .umap_ import UMAP

if TYPE_CHECKING:
    from polars_ml.pipeline.pipeline import Pipeline


class ReductionNameSpace:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def pca(
        self,
        components: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **pca_params: Any,
    ) -> Pipeline:
        return self.pipeline.pipe(PCA(components, features, **pca_params))

    def umap(
        self,
        components: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **umap_params: Any,
    ) -> Pipeline:
        return self.pipeline.pipe(UMAP(components, features, **umap_params))
