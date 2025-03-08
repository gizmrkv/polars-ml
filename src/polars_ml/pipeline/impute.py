from typing import Mapping, Self

import polars as pl
from polars import DataFrame

from polars_ml.pipeline.component import PipelineComponent


class Impute(PipelineComponent):
    def __init__(
        self,
        imputer: PipelineComponent,
        column: str,
        *,
        maintain_order: bool = False,
    ):
        self.imputer = imputer
        self.column = column
        self.maintain_order = maintain_order

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        train_data = data.filter(pl.col(self.column).is_not_null())
        if isinstance(validation_data, DataFrame):
            validation_data = validation_data.filter(pl.col(self.column).is_not_null())
        elif isinstance(validation_data, Mapping):
            validation_data = {
                key: value.filter(pl.col(self.column).is_not_null())
                for key, value in validation_data.items()
            }

        self.imputer.fit(train_data, validation_data)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        if self.maintain_order:
            data = data.with_row_index("index")
            missing_data = data.filter(pl.col(self.column).is_null())
            imputed_data = self.imputer.transform(
                missing_data.drop(self.column, "index")
            )
            filled_data = missing_data.with_columns(imputed_data[self.column])
            data = pl.concat(
                [data.filter(pl.col(self.column).is_not_null()), filled_data]
            )
            return data.sort("index").drop("index")
        else:
            missing_data = data.filter(pl.col(self.column).is_null())
            imputed_data = self.imputer.transform(missing_data.drop(self.column))
            filled_data = missing_data.with_columns(imputed_data[self.column])
            data = pl.concat(
                [data.filter(pl.col(self.column).is_not_null()), filled_data]
            )
            return data

    def fit_transform(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> DataFrame:
        if isinstance(validation_data, DataFrame):
            validation_data = validation_data.filter(pl.col(self.column).is_not_null())
        elif isinstance(validation_data, Mapping):
            validation_data = {
                key: value.filter(pl.col(self.column).is_not_null())
                for key, value in validation_data.items()
            }

        if self.maintain_order:
            data = data.with_row_index("index")
            missing_data = data.filter(pl.col(self.column).is_null())
            imputed_data = self.imputer.fit_transform(
                missing_data.drop(self.column, "index"), validation_data
            )
            filled_data = missing_data.with_columns(imputed_data[self.column])
            data = pl.concat(
                [data.filter(pl.col(self.column).is_not_null()), filled_data]
            )
            return data.sort("index").drop("index")
        else:
            missing_data = data.filter(pl.col(self.column).is_null())
            imputed_data = self.imputer.fit_transform(
                missing_data.drop(self.column), validation_data
            )
            filled_data = missing_data.with_columns(imputed_data[self.column])
            data = pl.concat(
                [data.filter(pl.col(self.column).is_not_null()), filled_data]
            )
            return data
