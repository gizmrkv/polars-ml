from __future__ import annotations

from typing import Any, Self

import polars as pl

from polars_ml import LazyTransformer


class LazyGetAttr(LazyTransformer):
    def __init__(self, attr: str, obj: Any | None, *args: Any, **kwargs: Any) -> None:
        self.attr = attr
        self.obj = obj
        self.args = args
        self.kwargs = kwargs

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        for arg in self.args:
            if isinstance(arg, LazyTransformer):
                arg.fit(data, **more_data)
        for arg in self.kwargs.values():
            if isinstance(arg, LazyTransformer):
                arg.fit(data, **more_data)
        return self

    def fit_transform(
        self, data: pl.DataFrame, **more_data: pl.DataFrame
    ) -> pl.DataFrame:
        args = [
            arg.fit_transform(data, **more_data)
            if isinstance(arg, LazyTransformer)
            else arg
            for arg in self.args
        ]
        kwargs = {
            key: val.fit_transform(data, **more_data)
            if isinstance(val, LazyTransformer)
            else val
            for key, val in self.kwargs.items()
        }
        obj = data if self.obj is None else self.obj
        output = getattr(obj, self.attr)(*args, **kwargs)
        return output if isinstance(output, pl.DataFrame) else data

    def transform(self, data: pl.LazyFrame) -> pl.LazyFrame:
        args = [
            arg.transform(data) if isinstance(arg, LazyTransformer) else arg
            for arg in self.args
        ]
        kwargs = {
            key: val.transform(data) if isinstance(val, LazyTransformer) else val
            for key, val in self.kwargs.items()
        }
        obj = data if self.obj is None else self.obj
        output = getattr(obj, self.attr)(*args, **kwargs)
        return output if isinstance(output, pl.LazyFrame) else data
