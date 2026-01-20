import uuid
from typing import Iterable, Iterator

import polars as pl
from polars import DataFrame, Series
from polars._typing import IntoExpr


class KFold:
    def __init__(
        self,
        n_splits: int,
        *,
        shuffle: bool = False,
        seed: int | None = None,
        stratify: IntoExpr | Iterable[IntoExpr] | None = None,
    ):
        self._n_splits = n_splits
        self._shuffle = shuffle
        self._seed = seed
        self._stratify = stratify

        self._fold_name = self.__class__.__name__ + "_" + uuid.uuid4().hex[:8]
        self._index_name = self.__class__.__name__ + "_" + uuid.uuid4().hex[:8]

        self._fold_expr = pl.cum_count(self._fold_name) - 1
        if shuffle:
            self._fold_expr = self._fold_expr.shuffle(seed=seed)
        if stratify is not None:
            self._fold_expr = self._fold_expr.over(stratify)

    def split(self, data: DataFrame) -> Iterator[tuple[Series, Series]]:
        data = data.with_columns(pl.lit(1).alias(self._fold_name))
        fold = data.select(self._fold_expr % self._n_splits).with_row_index(
            self._index_name
        )
        for i in range(self._n_splits):
            yield (
                fold.filter(pl.col(self._fold_name) != i)[self._index_name],
                fold.filter(pl.col(self._fold_name) == i)[self._index_name],
            )
