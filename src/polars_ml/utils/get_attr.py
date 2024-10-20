from typing import Any, override

import polars as pl
from polars import DataFrame, LazyFrame

from polars_ml.component import Component, LazyComponent


class GetAttr(Component):
    def __init__(self, method: str, *args: Any, **kwargs: Any):
        self.method = method
        self.args = args
        self.kwargs = kwargs
        self._is_fitted = True

        self.set_component_name(
            "".join(w[0].capitalize() + w[1:] for w in method.split("_"))
        )

    @override
    def execute(self, data: DataFrame) -> DataFrame:
        return getattr(data, self.method)(*self.args, **self.kwargs)


class LazyGetAttr(LazyComponent):
    def __init__(self, method: str, *args: Any, **kwargs: Any):
        self.method = method
        self.args = args
        self.kwargs = kwargs
        self._is_fitted = True

        self.set_component_name(
            "".join(w[0].capitalize() + w[1:] for w in method.split("_"))
        )

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        return getattr(data, self.method)(*self.args, **self.kwargs)


class GetAttrWithName(Component):
    def __init__(
        self, method: str, *args: Any, include_name: bool = False, **kwargs: Any
    ):
        self.method = method
        self.args = args
        self.kwargs = kwargs
        self.include_name = include_name
        self._is_fitted = True

        self.set_component_name(
            "".join(w[0].capitalize() + w[1:] for w in method.split("_"))
        )

    @override
    def execute(self, data: DataFrame) -> DataFrame:
        data = getattr(data, self.method)(*self.args, **self.kwargs)
        if self.include_name:
            data = data.select(pl.lit(self.method).alias("method"), pl.all())
        return data


class LazyGetAttrWithName(LazyComponent):
    def __init__(
        self, method: str, *args: Any, include_name: bool = False, **kwargs: Any
    ):
        self.method = method
        self.args = args
        self.kwargs = kwargs
        self.include_name = include_name
        self._is_fitted = True

        self.set_component_name(
            "".join(w[0].capitalize() + w[1:] for w in method.split("_"))
        )

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        data = getattr(data, self.method)(*self.args, **self.kwargs)
        if self.include_name:
            data = data.select(pl.lit(self.method).alias("method"), pl.all())
        return data
