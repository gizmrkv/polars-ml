from typing import Any, Self

import polars as pl
from polars import DataFrame

from polars_ml import Transformer

from .basic import Const


class GetAttr(Transformer):
    def __init__(self, attr: str, *args: Any, **kwargs: Any):
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    def transform(self, data: DataFrame) -> DataFrame:
        output = getattr(data, self.attr)(*self.args, **self.kwargs)
        return output if isinstance(output, DataFrame) else data


class LazyGetAttr(Transformer):
    def __init__(self, attr: str, *args: Any, **kwargs: Any):
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    def transform(self, data: DataFrame) -> DataFrame:
        output = getattr(data, self.attr)(*self.args, **self.kwargs)
        return output if isinstance(output, DataFrame) else data


class GetAttrOther(Transformer):
    def __init__(
        self,
        other: DataFrame | Transformer,
        attr: str,
        *args: Any,
        **kwargs: Any,
    ):
        self.other = Const(other).eager() if isinstance(other, DataFrame) else other
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self.other.fit(data, **more_data)
        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        other_df = self.other.fit_transform(data, **more_data)
        return getattr(data, self.attr)(other_df, *self.args, **self.kwargs)

    def transform(self, data: DataFrame) -> DataFrame:
        other_df = self.other.transform(data)
        return getattr(data, self.attr)(other_df, *self.args, **self.kwargs)


class LazyGetAttrOther(Transformer):
    def __init__(
        self,
        other: DataFrame | Transformer,
        attr: str,
        *args: Any,
        **kwargs: Any,
    ):
        self.other = Const(other) if isinstance(other, DataFrame) else other
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self.other.fit(data, **more_data)
        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        other_df = self.other.fit_transform(data, **more_data)
        return getattr(data, self.attr)(other_df, *self.args, **self.kwargs)

    def transform(self, data: DataFrame) -> DataFrame:
        other_df = self.other.transform(data)
        return getattr(data, self.attr)(other_df, *self.args, **self.kwargs)


class GetAttrPolars(Transformer):
    def __init__(self, attr: str, *args: Any, **kwargs: Any):
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    def transform(self, data: DataFrame) -> DataFrame:
        return getattr(pl, self.attr)(*self.args, **self.kwargs)


class LazyGetAttrPolars(Transformer):
    def __init__(self, attr: str, *args: Any, **kwargs: Any):
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    def transform(self, data: DataFrame) -> DataFrame:
        return getattr(pl, self.attr)(*self.args, **self.kwargs)
