from pathlib import Path
from typing import Any, Literal, override

from polars import DataFrame

from polars_ml.component import Component


class Write(Component):
    def __init__(
        self,
        file: str | Path,
        method: Literal["csv", "parquet", "avro"],
        *args: Any,
        **kwargs: Any,
    ):
        self.file = file
        self.method = method
        self.args = args
        self.kwargs = kwargs
        self._is_fitted = True

        self.set_component_name(
            "WriteCSV"
            if method == "csv"
            else "WriteParquet"
            if method == "parquet"
            else "WriteAvro"
        )

    @override
    def execute(self, data: DataFrame) -> DataFrame:
        if log_dir := self.log_dir:
            file = log_dir / self.file
            if self.method == "csv":
                data.write_csv(file, *self.args, **self.kwargs)
            if self.method == "parquet":
                data.write_parquet(file, *self.args, **self.kwargs)
            if self.method == "avro":
                data.write_avro(file, *self.args, **self.kwargs)

        return data
