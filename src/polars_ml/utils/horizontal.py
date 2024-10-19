import uuid
from typing import Dict, Iterable, override

from polars import LazyFrame
from polars._typing import IntoExpr

from polars_ml.component import LazyComponent


class LazyHorizontalAgg(LazyComponent):
    def __init__(
        self,
        *exprs: IntoExpr | Iterable[IntoExpr],
        variable_name: str | None = None,
        value_name: str | None = None,
        maintain_order: bool = True,
        aggs: Iterable[IntoExpr | Iterable[IntoExpr]] | None = None,
        named_aggs: Dict[str, IntoExpr] | None = None,
    ):
        self.exprs = exprs
        self.variable_name = variable_name
        self.value_name = value_name
        self.maintain_order = maintain_order
        self.aggs = aggs
        self.named_aggs = named_aggs
        self._is_fitted = True

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        index_name = uuid.uuid4().hex
        data = data.select(*self.exprs).with_row_index(index_name)
        return data.join(
            data.unpivot(
                data.collect_schema().names(),
                index=index_name,
                variable_name=self.variable_name,
                value_name=self.value_name,
            )
            .group_by(index_name, maintain_order=self.maintain_order)
            .agg(*(self.aggs or []), **(self.named_aggs or {})),
            index_name,
        ).drop(index_name)
