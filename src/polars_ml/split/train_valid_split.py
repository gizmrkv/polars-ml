import uuid
from typing import override

import polars as pl
from polars import LazyFrame

from polars_ml.component import LazyComponent


class TrainValidSplit(LazyComponent):
    def __init__(
        self,
        test_size: float,
        *,
        split_name: str = "is_valid",
        stratify: str | None = None,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        if not 0 <= test_size <= 1:
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

        self.test_size = test_size
        self.split_name = split_name
        self.stratify = stratify
        self.shuffle = shuffle
        self.seed = seed
        self._is_fitted = True

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        columns = data.collect_schema().names()
        height = int(data.select(columns[0]).collect().height)

        if self.stratify is None:
            split = (
                data.with_row_index(self.split_name)
                .with_columns(
                    pl.col(self.split_name).shuffle(seed=self.seed)
                    if self.shuffle
                    else pl.col(self.split_name)
                )
                .with_columns(pl.col(self.split_name) < self.test_size * height)
            )
        else:
            ratio = (
                data.group_by(self.stratify)
                .agg((pl.len() * self.test_size).ceil().alias(str(uuid.uuid4())))
                .collect()
            )

            split = (
                data.with_columns(
                    pl.cum_count(self.stratify)
                    .over(self.stratify)
                    .alias(self.split_name)
                )
                .with_columns(
                    pl.col(self.split_name).shuffle(seed=self.seed).over(self.stratify)
                    if self.shuffle
                    else pl.all()
                )
                .with_columns(
                    pl.col(self.split_name)
                    <= pl.col(self.stratify).replace_strict(
                        *ratio, return_dtype=pl.Float32
                    )
                )
            )

        return split
