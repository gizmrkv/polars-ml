import math
import uuid
from abc import ABC, abstractmethod
from typing import Mapping, Self

import polars as pl
from polars import DataFrame, Expr

from polars_ml import Component, utils


class Operator(Component, ABC):
    args: list["Operator"]
    name: str
    order: int

    def __init__(self, *args: "Operator", name: str | None = None):
        self.args = list(args)
        self.name = name or uuid.uuid4().hex
        self.order = sum(arg.order for arg in self.args)

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        for op in self.args:
            op.fit(data, validation_data)
        return self


class Column(Operator):
    def __init__(self, name: str):
        super().__init__(name=name)

    def transform(self, data: DataFrame) -> DataFrame:
        return data.select(self.name)

    def __str__(self) -> str:
        return self.name


class UnaryOperator(Operator, ABC):
    def __init__(self, arg: Operator, name: str | None = None):
        super().__init__(arg, name=name)

    @abstractmethod
    def op(self, arg: Expr) -> Expr: ...

    def transform(self, data: DataFrame) -> DataFrame:
        (arg,) = self.args
        return arg.transform(data).select(self.op(pl.col(arg.name)).alias(self.name))


class Log(UnaryOperator):
    def __init__(
        self,
        arg: Operator,
        name: str | None = None,
        base: float = math.e,
        plus: float = 0.0,
    ):
        super().__init__(arg, name=name)
        self.base = base
        self.plus = plus

    def op(self, arg: Expr) -> Expr:
        return (arg + self.plus).log(self.base)

    def __str__(self) -> str:
        log = "ln" if self.base == math.e else f"log{self.base}"
        return f"{log}({self.args[0]})"


class Sqrt(UnaryOperator):
    def op(self, arg: Expr) -> Expr:
        return arg.sqrt()

    def __str__(self) -> str:
        return f"sqrt({self.args[0]})"


class Sin(UnaryOperator):
    def op(self, arg: Expr) -> Expr:
        return arg.sin()

    def __str__(self) -> str:
        return f"sin({self.args[0]})"


class Cos(UnaryOperator):
    def op(self, arg: Expr) -> Expr:
        return arg.cos()

    def __str__(self) -> str:
        return f"cos({self.args[0]})"


class Tan(UnaryOperator):
    def op(self, arg: Expr) -> Expr:
        return arg.tan()

    def __str__(self) -> str:
        return f"tan({self.args[0]})"


class BinaryOperator(Operator, ABC):
    def __init__(self, lhs: Operator, rhs: Operator, name: str | None = None):
        super().__init__(lhs, rhs, name=name)

    @abstractmethod
    def op(self, lhs: Expr, rhs: Expr) -> Expr: ...

    def transform(self, data: DataFrame) -> DataFrame:
        lhs, rhs = self.args
        return pl.concat(
            [lhs.transform(data), rhs.transform(data)], how="horizontal"
        ).select(self.op(pl.col(lhs.name), pl.col(rhs.name)).alias(self.name))


class Add(BinaryOperator):
    def op(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs + rhs

    def __str__(self) -> str:
        return f"{self.args[0]} + {self.args[1]}"


class Sub(BinaryOperator):
    def op(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs - rhs

    def __str__(self) -> str:
        return f"{self.args[0]} - {self.args[1]}"


class Mul(BinaryOperator):
    def op(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs * rhs

    def __str__(self) -> str:
        return f"{self.args[0]} * {self.args[1]}"


class Div(BinaryOperator):
    def op(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs / rhs

    def __str__(self) -> str:
        return f"{self.args[0]} / {self.args[1]}"


class Max(BinaryOperator):
    def op(self, lhs: Expr, rhs: Expr) -> Expr:
        return pl.when(lhs > rhs).then(lhs).otherwise(rhs)

    def __str__(self) -> str:
        return f"max({self.args[0]}, {self.args[1]})"


class Min(BinaryOperator):
    def op(self, lhs: Expr, rhs: Expr) -> Expr:
        return pl.when(lhs < rhs).then(lhs).otherwise(rhs)

    def __str__(self) -> str:
        return f"min({self.args[0]}, {self.args[1]})"


class GroupByThen(Operator, ABC):
    def __init__(self, by: Operator, *aggs: Operator, name: str | None = None):
        super().__init__(by, *aggs, name=name)
        self.group_by = utils.GroupByThen(
            by.name,
            self.agg(*(pl.col(arg.name) for arg in self.args[1:])).alias(self.name),
        )

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        super().fit(data, validation_data)
        self.group_by.fit(
            pl.concat([arg.transform(data) for arg in self.args], how="horizontal")
        )
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        return self.group_by.transform(
            pl.concat([arg.transform(data) for arg in self.args], how="horizontal")
        ).select(self.name)

    @abstractmethod
    def agg(self, *aggs: Expr) -> Expr: ...


class GroupByLen(GroupByThen):
    def __init__(self, by: Operator, *, name: str | None = None):
        super().__init__(by, name=name)

    def agg(self, *aggs: Expr) -> Expr:
        return pl.len()

    def __str__(self) -> str:
        return f"len({self.args[0]})"


class GroupByMean(GroupByThen):
    def __init__(self, by: Operator, val: Operator, *, name: str | None = None):
        super().__init__(by, val, name=name)

    def agg(self, *aggs: Expr) -> Expr:
        return aggs[0].mean()

    def __str__(self) -> str:
        return f"mean({self.args[1]} over {self.args[0]})"


class GroupBySum(GroupByThen):
    def __init__(self, by: Operator, val: Operator, *, name: str | None = None):
        super().__init__(by, val, name=name)

    def agg(self, *aggs: Expr) -> Expr:
        return aggs[0].sum()

    def __str__(self) -> str:
        return f"sum({self.args[1]} over {self.args[0]})"


class GroupByStd(GroupByThen):
    def __init__(self, by: Operator, val: Operator, *, name: str | None = None):
        super().__init__(by, val, name=name)

    def agg(self, *aggs: Expr) -> Expr:
        return aggs[0].std()

    def __str__(self) -> str:
        return f"std({self.args[1]} over {self.args[0]})"


class GroupByNUnique(GroupByThen):
    def __init__(self, by: Operator, val: Operator, *, name: str | None = None):
        super().__init__(by, val, name=name)

    def agg(self, *aggs: Expr) -> Expr:
        return aggs[0].n_unique()

    def __str__(self) -> str:
        return f"n_unique({self.args[1]} over {self.args[0]})"
