import uuid
from typing import Iterable, Iterator

import polars as pl
from polars import DataFrame, Series
from polars._typing import IntoExpr


class KFold:
    def __init__(
        self,
        *,
        n_splits: int,
        shuffle: bool = False,
        seed: int | None = None,
        stratify: IntoExpr | Iterable[IntoExpr] | None = None,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed
        self.stratify = stratify

        self.fold_name = uuid.uuid4().hex
        self.index_name = uuid.uuid4().hex

        self.fold_expr = pl.cum_count(self.fold_name) - 1
        if shuffle:
            self.fold_expr = self.fold_expr.shuffle(seed=seed)
        if stratify is not None:
            self.fold_expr = self.fold_expr.over(stratify)

    def split(self, data: DataFrame) -> Iterator[tuple[Series, Series]]:
        data = data.with_columns(pl.lit(1).alias(self.fold_name))
        fold = data.select(self.fold_expr % self.n_splits).with_row_index(
            self.index_name
        )
        for i in range(self.n_splits):
            yield (
                fold.filter(pl.col(self.fold_name) != i)[self.index_name],
                fold.filter(pl.col(self.fold_name) == i)[self.index_name],
            )
