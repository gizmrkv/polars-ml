from typing import Any, Callable

from polars import DataFrame

from .component import Component


class Apply(Component):
    def __init__(self, func: Callable[[DataFrame], DataFrame | Any]):
        self.func = func

    def transform(self, data: DataFrame) -> DataFrame:
        output = self.func(data)
        if isinstance(output, DataFrame):
            return output
        else:
            return data
