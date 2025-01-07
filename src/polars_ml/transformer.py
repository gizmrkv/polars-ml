from abc import ABC, abstractmethod
from typing import Iterable, Self

import polars as pl
import polars.selectors as cs
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml import Component


class MoveScaler(Component, ABC):
    def __init__(self, *expr: IntoExpr | Iterable[IntoExpr]):
        self.exprs = expr
        self.moves: dict[str, float] = {}
        self.scales: dict[str, float] = {}

    @abstractmethod
    def get_moves(self, data: DataFrame) -> dict[str, float]: ...

    @abstractmethod
    def get_scales(self, data: DataFrame) -> dict[str, float]: ...

    def fit(self, data: DataFrame) -> Self:
        data = data.select(*self.exprs)
        self.moves = self.get_moves(data)
        self.scales = self.get_scales(data)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        columns = (
            data.lazy()
            .select(cs.matches(r"|".join(self.moves.keys())))
            .collect_schema()
            .names()
        )
        return data.with_columns(
            [(pl.col(col) - self.moves[col]) / self.scales[col] for col in columns]
        )


class StandardScaler(MoveScaler):
    def get_moves(self, data: DataFrame) -> dict[str, float]:
        return data.mean().row(0, named=True)

    def get_scales(self, data: DataFrame) -> dict[str, float]:
        return data.std().row(0, named=True)


class MinMaxScaler(MoveScaler):
    def get_moves(self, data: DataFrame) -> dict[str, float]:
        return data.min().row(0, named=True)

    def get_scales(self, data: DataFrame) -> dict[str, float]:
        return data.select(pl.all().max() - pl.all().min()).row(0, named=True)


class QuantileScaler(MoveScaler):
    def __init__(
        self,
        *expr: IntoExpr | Iterable[IntoExpr],
        quantile: tuple[float, float] = (0.25, 0.75),
    ):
        super().__init__(*expr)
        self.quantile = quantile
        q1, q3 = quantile
        assert (
            0 <= q1 < q3 <= 1
        ), f"Quantile values must be in the range [0, 1] and q1 < q3. Got {quantile}"

    def get_moves(self, data: DataFrame) -> dict[str, float]:
        return data.quantile(0.5).row(0, named=True)

    def get_scales(self, data: DataFrame) -> dict[str, float]:
        q1, q3 = self.quantile
        return data.select(pl.all().quantile(q3) - pl.all().quantile(q1)).row(
            0, named=True
        )
