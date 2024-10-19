from typing import Literal, override

from polars import DataFrame, LazyFrame

from polars_ml.component import Component, LazyComponent


class Echo(LazyComponent):
    def __init__(self):
        self._is_fitted = True


class Lit(LazyComponent):
    def __init__(self, data: DataFrame):
        self.data = data
        self._is_fitted = True

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        return self.data.lazy()


class SortColumns(LazyComponent):
    def __init__(
        self, by: Literal["dtype", "name"] = "dtype", *, descending: bool = False
    ):
        self.by = by
        self.descending = descending
        self._is_fitted = True

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        schema = data.collect_schema()
        sorted_columns = sorted(
            [{"name": k, "dtype": str(v) + k} for k, v in schema.items()],
            key=lambda x: x[self.by],
            reverse=self.descending,
        )
        return data.select([col["name"] for col in sorted_columns])


class Print(Component):
    def __init__(self):
        self._is_fitted = True

    @override
    def execute(self, data: DataFrame) -> DataFrame:
        print(data)
        return data


class Display(Component):
    def __init__(self):
        self._is_fitted = True

    @override
    def execute(self, data: DataFrame) -> DataFrame:
        from IPython.display import display  # type: ignore

        display(data)
        return data
