from __future__ import annotations

from typing import Any, Self

import polars as pl
from polars import DataFrame

from polars_ml.base import Transformer


class GetAttr(Transformer):
    def __init__(self, attr: str, obj: Any | None, *args: Any, **kwargs: Any):
        self.attr = attr
        self.obj = obj
        self.args = args
        self.kwargs = kwargs

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        for arg in self.args:
            if isinstance(arg, Transformer):
                arg.fit(data, **more_data)
        for arg in self.kwargs.values():
            if isinstance(arg, Transformer):
                arg.fit(data, **more_data)
        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        args = [
            arg.fit_transform(data, **more_data)
            if isinstance(arg, Transformer)
            else arg
            for arg in self.args
        ]
        kwargs = {
            key: val.fit_transform(data, **more_data)
            if isinstance(val, Transformer)
            else val
            for key, val in self.kwargs.items()
        }
        obj = data if self.obj is None else self.obj
        output = getattr(obj, self.attr)(*args, **kwargs)
        return output if isinstance(output, DataFrame) else data

    def transform(self, data: DataFrame) -> DataFrame:
        args = [
            arg.transform(data) if isinstance(arg, Transformer) else arg
            for arg in self.args
        ]
        kwargs = {
            key: val.transform(data) if isinstance(val, Transformer) else val
            for key, val in self.kwargs.items()
        }
        obj = data if self.obj is None else self.obj
        output = getattr(obj, self.attr)(*args, **kwargs)
        return output if isinstance(output, DataFrame) else data
