import uuid
from typing import Any, Dict, Iterable, Self, Sequence, override

import polars as pl
import polars.selectors as cs
from polars import DataFrame, LazyFrame, Series
from polars._typing import IntoExpr

from polars_ml.component import LazyComponent


class OrdinalEncoder(LazyComponent):
    def __init__(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        orders: Dict[str, Sequence[Any]] | None = None,
        maintain_order: bool = False,
    ):
        self.exprs = exprs
        self.orders = orders or {}
        self.maintain_order = maintain_order
        self.order_name = uuid.uuid4().hex

    @override
    def fit(self, data: LazyFrame) -> Self:
        data = data.select(*self.exprs)

        self.mappings = {
            col: DataFrame(
                [
                    Series(col, self.orders[col]),
                    Series(
                        self.order_name, range(len(self.orders[col])), dtype=pl.UInt32
                    ),
                ]
            )
            for col in self.orders
        }

        for col in data.collect_schema().names() + list(self.orders.keys()):
            if col in self.orders:
                continue

            self.mappings[col] = (
                data.select(col)
                .unique(maintain_order=self.maintain_order)
                .drop_nulls()
                .with_row_index(self.order_name)
                .collect()
            )

        if log_dir := self.log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            for col, map in self.mappings.items():
                map.rename({col: "column", self.order_name: "label"}).write_csv(
                    log_dir / f"{col}.csv"
                )

        return self

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        data_new = data.select(*self.exprs)
        cols_new = data_new.collect_schema().names()
        cols_old = data.collect_schema().names()
        cols_add = [col for col in cols_new if col not in cols_old]
        cols_upd = [col for col in cols_new if col in cols_old]

        data_new = pl.concat(
            [
                data_new.join(map.lazy(), on=col, how="left").select(
                    pl.col(self.order_name).alias(col)
                )
                for col, map in self.mappings.items()
            ],
            how="horizontal",
        )
        index_name = uuid.uuid4().hex

        data = (
            data.with_row_index(index_name)
            .drop(cs.by_name(cols_upd))
            .join(data_new.with_row_index(index_name), on=index_name)
            .select(cols_old)
        )
        data = pl.concat([data, data_new.select(cols_add)], how="horizontal")
        return data
