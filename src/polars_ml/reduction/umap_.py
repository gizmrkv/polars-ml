from __future__ import annotations

from typing import Any, Iterable, Self, Sequence

import polars as pl
import polars.selectors as cs
from polars._typing import ColumnNameOrSelector

from polars_ml.base import Transformer
from polars_ml.exceptions import NotFittedError


class UMAP(Transformer):
    def __init__(
        self,
        components: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **umap_params: Any,
    ):
        self._components = (
            [components] if isinstance(components, str) else list(components)
        )
        self._features_selector = features if features is not None else cs.all()
        self._umap_params = umap_params

        self._features: list[str] | None = None
        self._estimator: Any = None

    @property
    def features(self) -> list[str]:
        if self._features is None:
            raise NotFittedError()
        return self._features

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        import umap

        self._features = (
            data.lazy().select(self._features_selector).collect_schema().names()
        )

        X = data.select(*self.features).to_numpy()

        self._estimator = umap.UMAP(**self._umap_params)
        self._estimator.fit(X)

        return self

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        if self._estimator is None or self._features is None:
            raise NotFittedError()

        X = data.select(*self.features).to_numpy()
        pred = self._estimator.transform(X)
        return pl.from_numpy(pred, schema=self._components)
