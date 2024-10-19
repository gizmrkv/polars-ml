import uuid
from typing import Iterable, Literal, Self, Tuple, override

import polars as pl
import polars.selectors as cs
from polars import DataFrame, LazyFrame
from polars._typing import IntoExpr

from polars_ml.component import LazyComponent


class MoveScaler(LazyComponent):
    def __init__(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        method: Literal["min_max", "standard", "robust"] = "min_max",
        quantile: Tuple[float, float] = (0.25, 0.75),
    ):
        self.exprs = exprs
        self.method = method
        self.quantile = quantile

    @override
    def fit(self, data: LazyFrame) -> Self:
        data = data.select(*self.exprs)
        if self.method == "min_max":
            move = data.min()
            scale = data.select(pl.all().max() - pl.all().min())
        elif self.method == "standard":
            move = data.mean()
            scale = data.std()
        elif self.method == "robust":
            q1, q3 = self.quantile
            assert (
                0 <= q1 < q3 <= 1
            ), "Quantile values should be in the range [0, 1] and q1 < q3"
            move = data.median()
            scale = data.select(pl.all().quantile(q3) - pl.all().quantile(q1))
        else:
            raise ValueError(f"Unknown method {self.method}")

        self.move = move.collect().row(0, named=True)
        self.scale = scale.collect().row(0, named=True)

        if log_dir := self.log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            DataFrame(
                dict(
                    [
                        ("column", self.move.keys()),
                        ("move", self.move.values()),
                        ("scale", self.scale.values()),
                    ]
                )
            ).write_csv(log_dir / "move_scale.csv")

        return self

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        data_new = data.select(*self.exprs)
        cols_new = data_new.collect_schema().names()
        cols_old = data.collect_schema().names()
        cols_add = [col for col in cols_new if col not in cols_old]
        cols_upd = [col for col in cols_new if col in cols_old]

        data_new = data_new.with_columns(
            [(pl.col(col) - self.move[col]) / (self.scale[col]) for col in cols_new]
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
