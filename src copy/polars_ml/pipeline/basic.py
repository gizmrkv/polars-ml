from typing import Callable, Generic, Self, TypeVar

from polars import DataFrame
from polars._typing import EngineType
from polars.DataFrame import QueryOptFlags
from polars.DataFrame.opt_flags import DEFAULT_QUERY_OPT_FLAGS

from polars_ml import Transformer


class Apply(Transformer):
    def __init__(self, func: Callable[[DataFrame], DataFrame]):
        self.func = func

    def transform(self, data: DataFrame) -> DataFrame:
        return self.func(data)


class LazyApply(Transformer):
    def __init__(self, func: Callable[[DataFrame], DataFrame]):
        self.func = func

    def transform(self, data: DataFrame) -> DataFrame:
        return self.func(data)


class Echo(Transformer):
    def transform(self, data: DataFrame) -> DataFrame:
        return data


class Const(Transformer):
    def __init__(self, data: DataFrame | DataFrame):
        self.data = data.lazy() if isinstance(data, DataFrame) else data

    def transform(self, data: DataFrame) -> DataFrame:
        return self.data


class Parrot(Transformer):
    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self.data = data
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        return self.data.lazy()


class Collect(Transformer):
    def __init__(
        self,
        *,
        engine: EngineType = "auto",
        background: bool = False,
        optimizations: QueryOptFlags = DEFAULT_QUERY_OPT_FLAGS,
    ):
        self.params = {
            "engine": engine,
            "background": background,
            "optimizations": optimizations,
        }

    def transform(self, data: DataFrame) -> DataFrame:
        return data.collect(**self.params).lazy()


TransformerType = TypeVar("TransformerType", bound=Transformer)
TransformerType = TypeVar("TransformerType", bound=Transformer)


class Side(Transformer, Generic[TransformerType]):
    def __init__(self, Transformer: TransformerType):
        self.Transformer = Transformer

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self.Transformer.fit(data, **more_data)
        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        self.Transformer.fit_transform(data, **more_data)
        return data

    def transform(self, data: DataFrame) -> DataFrame:
        self.Transformer.transform(data)
        return data


class LazySide(Transformer, Generic[TransformerType]):
    def __init__(self, Transformer: TransformerType):
        self.Transformer = Transformer

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self.Transformer.fit(data, **more_data)
        return self

    def fit_transform(self, data: DataFrame, **more_data: DataFrame) -> DataFrame:
        self.Transformer.fit_transform(data, **more_data)
        return data

    def transform(self, data: DataFrame) -> DataFrame:
        self.Transformer.transform(data)
        return data
