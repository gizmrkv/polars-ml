from __future__ import annotations

from typing import Sequence

import polars as pl
from polars._typing import ColumnNameOrSelector

from polars_ml.base import Transformer


class StratifySample(Transformer):
    def __init__(
        self,
        by: ColumnNameOrSelector | Sequence[ColumnNameOrSelector],
        fraction: float,
        *,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
        maintain_order: bool = False,
    ):
        self._by = by
        self._fraction = fraction
        self._with_replacement = with_replacement
        self._shuffle = shuffle
        self._seed = seed
        self._maintain_order = maintain_order

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        sample_list = []
        by_columns = data.lazy().select(self._by).collect_schema().names()
        for by_data in data.partition_by(
            by_columns, maintain_order=self._maintain_order
        ):
            sample_list.append(
                by_data.sample(
                    fraction=self._fraction,
                    with_replacement=self._with_replacement,
                    shuffle=self._shuffle,
                    seed=self._seed,
                )
            )

        return pl.concat(sample_list)
