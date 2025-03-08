import itertools

import polars as pl
from polars import DataFrame
from polars._typing import ColumnNameOrSelector

from polars_ml.pipeline.component import PipelineComponent


class Polynomial(PipelineComponent):
    def __init__(
        self,
        *features: ColumnNameOrSelector,
        degree: int = 2,
    ):
        self.features = features
        self.degree = degree

    def transform(self, data: DataFrame) -> DataFrame:
        columns = data.lazy().select(*self.features).collect_schema().names()
        for a, b in itertools.combinations_with_replacement(columns, r=self.degree):
            data = data.with_columns((pl.col(a) * pl.col(b)).alias(f"{a} * {b}"))

        return data
