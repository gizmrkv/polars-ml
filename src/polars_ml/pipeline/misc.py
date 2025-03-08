from typing import Any, Literal

from polars import DataFrame

from polars_ml.pipeline.component import PipelineComponent


class Echo(PipelineComponent):
    def transform(self, data: DataFrame) -> DataFrame:
        return data


class GetAttr(PipelineComponent):
    def __init__(self, method: str, *args: Any, **kwargs: Any):
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def transform(self, data: DataFrame) -> DataFrame:
        return getattr(data, self.method)(*self.args, **self.kwargs)


class Print(PipelineComponent):
    def transform(self, data: DataFrame) -> DataFrame:
        print(data)
        return data


class Display(PipelineComponent):
    def transform(self, data: DataFrame) -> DataFrame:
        from IPython.display import display

        display(data)
        return data


class SortColumns(PipelineComponent):
    def __init__(
        self, by: Literal["dtype", "name"] = "dtype", *, descending: bool = False
    ):
        self.by = by
        self.descending = descending

    def transform(self, data: DataFrame) -> DataFrame:
        schema = data.collect_schema()
        sorted_columns = sorted(
            [{"name": k, "dtype": str(v) + k} for k, v in schema.items()],
            key=lambda x: x[self.by],
            reverse=self.descending,
        )
        return data.select([col["name"] for col in sorted_columns])
