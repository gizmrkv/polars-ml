from typing import Iterable, Self, override

from polars import LazyFrame
from polars._typing import ColumnNameOrSelector, IntoExpr

from polars_ml.component import LazyComponent


class TargetEncoder(LazyComponent):
    def __init__(
        self,
        by: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ):
        self.by = by
        self.aggs = aggs
        self.named_aggs = named_aggs

    @override
    def fit(self, data: LazyFrame) -> Self:
        self.mapping = (
            data.group_by(self.by).agg(*self.aggs, **self.named_aggs).collect()
        )
        if log_dir := self.log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.mapping.write_csv(log_dir / "mapping.csv")

        self._is_fitted = True
        return self

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        return data.join(
            self.mapping.lazy(),
            on=data.select(self.by).collect_schema().names(),
            how="left",
        )
