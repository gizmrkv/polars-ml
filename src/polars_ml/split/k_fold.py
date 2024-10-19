from typing import Iterable, override

import polars as pl
from polars import LazyFrame
from polars._typing import IntoExpr

from polars_ml.component import LazyComponent


class KFold(LazyComponent):
    def __init__(
        self,
        n_splits: int = 5,
        *,
        split_name: str = "fold",
        stratify: IntoExpr | Iterable[IntoExpr] | None = None,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        self.n_splits = n_splits
        self.split_name = split_name
        self.stratify = stratify
        self.shuffle = shuffle
        self.seed = seed
        self._is_fitted = True

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        split_expr = pl.col(self.split_name).cum_count() - 1
        if self.shuffle:
            split_expr = split_expr.shuffle(seed=self.seed)
        if self.stratify is not None:
            split_expr = split_expr.over(self.stratify)

        return data.with_columns(pl.lit(False).alias(self.split_name)).with_columns(
            split_expr % self.n_splits
        )
