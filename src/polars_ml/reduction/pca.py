from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Self, Sequence

import polars as pl
import polars.selectors as cs
from polars import DataFrame
from polars._typing import ColumnNameOrSelector

from polars_ml.base import Transformer
from polars_ml.exceptions import NotFittedError

if TYPE_CHECKING:
    from sklearn.decomposition import PCA as SKLearnPCA


class PCA(Transformer):
    def __init__(
        self,
        components: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        **pca_params: Any,
    ):
        self._components = (
            [components] if isinstance(components, str) else list(components)
        )
        self._features_selector = features if features is not None else cs.all()
        self._pca_params = pca_params

        self._features: list[str] | None = None
        self._estimator: SKLearnPCA | None = None

    @property
    def features(self) -> list[str]:
        if self._features is None:
            raise NotFittedError()
        return self._features

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        from sklearn.decomposition import PCA as SKLearnPCA

        self._features = (
            data.lazy().select(self._features_selector).collect_schema().names()
        )

        X = data.select(*self.features).to_numpy()

        self._estimator = SKLearnPCA(**self._pca_params)
        self._estimator.fit(X)

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if self._estimator is None or self._features is None:
            raise NotFittedError()

        X = data.select(*self.features).to_numpy()
        pred = self._estimator.transform(X)
        return pl.from_numpy(pred, schema=self._components)
