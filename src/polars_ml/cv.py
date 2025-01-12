import datetime
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
        stratify: IntoExpr | Iterable[IntoExpr],
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


class date_series_split:
    def __init__(
        self,
        data: DataFrame,
        date: str,
        *,
        initial_days: int,
        period_days: int,
        horizon_days: int,
        purged_days: int = 0,
        embarked_days: int | None = None,
    ):
        self.date = date
        self.initial_days = initial_days
        self.period_days = period_days
        self.horizon_days = horizon_days
        self.purged_days = purged_days
        self.embarked_days = embarked_days

        self.data = data.select(date).with_row_index("index")

        self.start_date: datetime.date = data[date].min()  # type: ignore
        self.end_date: datetime.date = data[date].max()  # type: ignore
        assert isinstance(self.start_date, datetime.date)
        assert isinstance(self.end_date, datetime.date)

    def __len__(self) -> int:
        return (
            (self.end_date - self.start_date).days
            - self.initial_days
            - self.purged_days
            - self.horizon_days
        ) // self.period_days

    def __iter__(self) -> Iterator[tuple[Series, Series]]:
        for i in range(len(self)):
            cutoff_date = self.start_date + datetime.timedelta(
                days=self.initial_days + i * self.period_days
            )
            train_idx = self.data.filter(
                (pl.col(self.date) < cutoff_date)
                | (
                    pl.col(self.date)
                    >= cutoff_date
                    + datetime.timedelta(
                        days=self.purged_days + self.horizon_days + self.embarked_days
                    )
                    if self.embarked_days is not None
                    else False
                )
            )["index"]
            valid_idx = self.data.filter(
                (
                    pl.col(self.date)
                    >= cutoff_date + datetime.timedelta(days=self.purged_days)
                )
                & (
                    pl.col(self.date)
                    < cutoff_date
                    + datetime.timedelta(days=self.purged_days + self.horizon_days)
                )
            )["index"]
            yield train_idx, valid_idx
