from typing import Iterable, Iterator

import polars as pl
from polars import DataFrame, Series
from polars._typing import IntoExpr


class k_fold:
    def __init__(
        self,
        data: DataFrame,
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

        data = data.with_columns(pl.lit(1).alias("fold"))

        fold_expr = pl.cum_count("fold") - 1
        if shuffle:
            fold_expr = fold_expr.shuffle(seed=seed)
        if stratify is not None:
            fold_expr = fold_expr.over(stratify)

        self.fold = data.select(fold_expr % n_splits).with_row_index("index")

    def __len__(self):
        return self.n_splits

    def __iter__(self) -> Iterator[tuple[Series, Series]]:
        for i in range(self.n_splits):
            yield (
                self.fold.filter(pl.col("fold") != i)["index"],
                self.fold.filter(pl.col("fold") == i)["index"],
            )
