from typing import Any, Iterable, Protocol, Self, override

import polars as pl
from numpy.typing import ArrayLike, NDArray
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.component import Component


class TransformerMixin(Protocol):
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self: ...
    def transform(self, X: ArrayLike) -> NDArray[Any]: ...
    def fit_transform(
        self, X: ArrayLike, y: NDArray[Any] | None = None
    ) -> ArrayLike: ...


class Transformer(Component):
    def __init__(
        self,
        transformer: TransformerMixin,
        *,
        X: IntoExpr | Iterable[IntoExpr],
        y: IntoExpr | None = None,
        name: str = "out",
    ):
        self.transformer = transformer
        self.X = X
        self.y = y
        self.name = name

    @override
    def fit(self, data: DataFrame) -> Self:
        X = data.select(self.X)
        X_np = X.to_numpy()
        y_np = data.select(self.y).to_numpy() if self.y is not None else None
        self.transformer.fit(X_np, y_np)
        self.X_columns = X.columns
        return self

    @override
    def execute(self, data: DataFrame) -> DataFrame:
        X_np = data.select(self.X).select(self.X_columns).to_numpy()
        y_np = self.transformer.transform(X_np)
        if y_np.ndim == 2 and y_np.shape[1] > 1:
            schema = [f"{self.name}_{i}" for i in range(y_np.shape[1])]
        else:
            schema = [self.name]

        out = pl.from_numpy(y_np, schema=schema)
        return pl.concat([data, out], how="horizontal")
