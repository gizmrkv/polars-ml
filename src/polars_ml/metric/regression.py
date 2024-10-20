from typing import Iterable, override

import polars as pl
from polars import LazyFrame
from polars._typing import IntoExpr

from polars_ml.component import LazyComponent


class R2(LazyComponent):
    def __init__(
        self,
        preds: IntoExpr | Iterable[IntoExpr],
        label: str,
        *,
        include_name: bool = False,
    ):
        self.preds = preds
        self.label = label
        self.include_name = include_name
        self._is_fitted = True

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        data = data.select(self.preds, self.label).select(
            1
            - (pl.col(self.label) - pl.all().exclude(self.label)).pow(2).sum()
            / (pl.col(self.label) - pl.col(self.label).mean()).pow(2).sum()
        )
        if self.include_name:
            data = data.select(pl.lit("r2").alias("method"), pl.all())

        return data
