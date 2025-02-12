import math
import uuid
from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Self

import polars as pl
from polars import DataFrame, Expr
from polars._typing import IntoExpr

from polars_ml import Component, group_by, preprocessing


class Operator(Component, ABC):
    args: list["Operator"]
    order: int = 0

    def __init__(self, *args: "Operator", name: str | None = None):
        self.args = list(args)
        self._name = name or uuid.uuid4().hex

        if len(self.args) == 0:
            self.order = 0
        else:
            self.order = 1 + max(arg.order for arg in self.args)

    @property
    def name(self) -> str:
        return self._name

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        for op in self.args:
            op.fit(data, validation_data)
        return self

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Operator):
            return False
        return self._name == other._name

    def __hash__(self) -> int:
        return hash(self._name)

    @abstractmethod
    def __str__(self) -> str: ...


class Column(Operator):
    def __init__(self, name: str):
        super().__init__(name=name)

    def transform(self, data: DataFrame) -> DataFrame:
        return data.select(self.name)

    def __str__(self) -> str:
        return self.name


class UnaryOperator(Operator, ABC):
    symbol: str

    def __init__(self, arg: Operator, name: str | None = None):
        super().__init__(arg, name=name)

    @abstractmethod
    def op(self, arg: Expr) -> Expr: ...

    def transform(self, data: DataFrame) -> DataFrame:
        (arg,) = self.args
        return arg.transform(data).select(self.op(pl.col(arg.name)).alias(self.name))

    def __str__(self) -> str:
        return f"{self.symbol}({self.args[0]})"


class Inv(UnaryOperator):
    symbol = "1/"

    def op(self, arg: Expr) -> Expr:
        return 1 / arg


class Pow(UnaryOperator):
    def __init__(self, arg: Operator, name: str | None = None, exp: float = math.e):
        super().__init__(arg, name=name)
        self.exp = exp
        self.symbol = f"pow{exp}" if exp != math.e else "exp"

    def op(self, arg: Expr) -> Expr:
        return arg.pow(self.exp)


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
        self.symbol = f"log{base}" if base != math.e else "ln"

    def op(self, arg: Expr) -> Expr:
        return (arg + self.plus).log(self.base)


class Sqrt(UnaryOperator):
    symbol = "sqrt"

    def op(self, arg: Expr) -> Expr:
        return arg.sqrt()


class Square(UnaryOperator):
    def op(self, arg: Expr) -> Expr:
        return arg**2

    def __str__(self) -> str:
        return f"({self.args[0]})**2"


class Sigmoid(UnaryOperator):
    symbol = "sigmoid"

    def op(self, arg: Expr) -> Expr:
        return 1 / (1 + (-arg).exp())


class Floor(UnaryOperator):
    symbol = "round"

    def op(self, arg: Expr) -> Expr:
        return arg.floor()


class Residual(UnaryOperator):
    symbol = "residual"

    def op(self, arg: Expr) -> Expr:
        return arg - arg.floor()


class Abs(UnaryOperator):
    symbol = "abs"

    def op(self, arg: Expr) -> Expr:
        return arg.abs()


class BinaryOperator(Operator, ABC):
    symbol: str

    def __init__(self, lhs: Operator, rhs: Operator, name: str | None = None):
        super().__init__(lhs, rhs, name=name)

    @abstractmethod
    def op(self, lhs: Expr, rhs: Expr) -> Expr: ...

    def transform(self, data: DataFrame) -> DataFrame:
        lhs, rhs = self.args
        return pl.concat(
            [lhs.transform(data), rhs.transform(data)], how="horizontal"
        ).select(self.op(pl.col(lhs.name), pl.col(rhs.name)).alias(self.name))

    def __str__(self) -> str:
        return f"{self.symbol}({self.args[0]}, {self.args[1]})"


class Add(BinaryOperator):
    symbol = "+"

    def op(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs + rhs


class Sub(BinaryOperator):
    symbol = "-"

    def op(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs - rhs


class Mul(BinaryOperator):
    symbol = "*"

    def op(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs * rhs


class Div(BinaryOperator):
    symbol = "/"

    def op(self, lhs: Expr, rhs: Expr) -> Expr:
        return lhs / rhs


class Max(BinaryOperator):
    symbol = "max"

    def op(self, lhs: Expr, rhs: Expr) -> Expr:
        return pl.when(lhs > rhs).then(lhs).otherwise(rhs)


class Min(BinaryOperator):
    symbol = "min"

    def op(self, lhs: Expr, rhs: Expr) -> Expr:
        return pl.when(lhs < rhs).then(lhs).otherwise(rhs)


class GroupByFreq(Operator):
    def __init__(
        self,
        by: Operator,
        *,
        name: str | None = None,
        probability: bool = False,
    ):
        super().__init__(by, name=name)
        self.probability = probability
        self.group_by = group_by.GroupByThen(
            by.name,
            pl.len().alias(self.name),
            maintain_order=True,
            after_with_columns=(pl.col(self.name) / pl.col(self.name).sum())
            if probability
            else None,
        )

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        super().fit(data, validation_data)
        self.group_by.fit(self.args[0].fit_transform(data, validation_data))
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        return self.group_by.transform(self.args[0].transform(data)).select(self.name)

    def __str__(self) -> str:
        return ("prob" if self.probability else "len") + f"({self.args[0]})"


class GroupByThen(Operator, ABC):
    symbol: str

    def __init__(
        self,
        by: Operator,
        *aggs: Operator,
        name: str | None = None,
        after_with_columns: IntoExpr | Iterable[IntoExpr] | None = None,
    ):
        super().__init__(by, *aggs, name=name)
        self.after_with_columns = after_with_columns
        self.group_by = group_by.GroupByThen(
            by.name,
            self.agg(*(pl.col(arg.name) for arg in self.args[1:])).alias(self.name),
            maintain_order=True,
            after_with_columns=after_with_columns,
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

    def __str__(self) -> str:
        return f"{self.symbol}({self.args[1]} over {self.args[0]})"


class GroupByMean(GroupByThen):
    symbol = "mean"

    def __init__(self, by: Operator, val: Operator, *, name: str | None = None):
        super().__init__(by, val, name=name)

    def agg(self, *aggs: Expr) -> Expr:
        return aggs[0].mean()


class GroupByStd(GroupByThen):
    symbol = "std"

    def __init__(self, by: Operator, val: Operator, *, name: str | None = None):
        super().__init__(by, val, name=name)

    def agg(self, *aggs: Expr) -> Expr:
        return aggs[0].std()


class GroupByMin(GroupByThen):
    symbol = "min"

    def __init__(self, by: Operator, val: Operator, *, name: str | None = None):
        super().__init__(by, val, name=name)

    def agg(self, *aggs: Expr) -> Expr:
        return aggs[0].min()


class GroupByMax(GroupByThen):
    symbol = "max"

    def __init__(self, by: Operator, val: Operator, *, name: str | None = None):
        super().__init__(by, val, name=name)

    def agg(self, *aggs: Expr) -> Expr:
        return aggs[0].max()


class GroupByQuantile(GroupByThen):
    def __init__(
        self,
        by: Operator,
        val: Operator,
        *,
        quantile: float = 0.5,
        name: str | None = None,
    ):
        self.quantile = quantile
        super().__init__(by, val, name=name)
        self.symbol = f"q{quantile}"

    def agg(self, *aggs: Expr) -> Expr:
        return aggs[0].quantile(self.quantile)


class GroupByNUnique(GroupByThen):
    symbol = "n_unique"

    def __init__(self, by: Operator, val: Operator, *, name: str | None = None):
        super().__init__(by, val, name=name)

    def agg(self, *aggs: Expr) -> Expr:
        return aggs[0].n_unique().cast(pl.Int64)


class Combine(Operator):
    def __init__(self, lhs: Operator, rhs: Operator, name: str | None = None):
        super().__init__(lhs, rhs, name=name)
        self.label_encoding = preprocessing.LabelEncoding(self.name)

    def combine(self, data: DataFrame) -> DataFrame:
        lhs, rhs = self.args
        return pl.concat(
            [lhs.transform(data), rhs.transform(data)], how="horizontal"
        ).select(
            pl.concat_str(
                pl.col(lhs.name).cast(pl.String),
                pl.col(rhs.name).cast(pl.String),
            ).alias(self.name)
        )

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        super().fit(data, validation_data)
        self.label_encoding.fit(self.combine(data))
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        return self.label_encoding.transform(self.combine(data))

    def __str__(self) -> str:
        return f"++({self.args[0]}, {self.args[1]})"
