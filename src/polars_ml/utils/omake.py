import uuid
from typing import Iterable

import polars as pl
from polars import Expr, LazyFrame
from polars._typing import ColumnNameOrSelector

from polars_ml.component import LazyComponent


def is_internal_point(x: str, y: str) -> Expr:
    return (
        (pl.col(x) - pl.col(x).shift(1)) * (pl.col(y).shift(-1) - pl.col(y))
        == (pl.col(x).shift(-1) - pl.col(x)) * (pl.col(y) - pl.col(y).shift(1))
    ).fill_null(False)


def trapezoid(x: str, y: str) -> Expr:
    return (
        (pl.col(x) - pl.col(x).shift(1)) * (pl.col(y) + pl.col(y).shift(1))
    ).sum() * 0.5


def recall(y_true: str, y_pred: str) -> Expr:
    return (pl.col(y_true) & pl.col(y_pred)).sum() / pl.col(y_true).sum()


def precision(y_true: str, y_pred: str) -> Expr:
    return (pl.col(y_true) & pl.col(y_pred)).sum() / pl.col(y_pred).sum()


def accuracy(y_true: str, y_pred: str) -> Expr:
    return (pl.col(y_true) == pl.col(y_pred)).mean()


def false_positive_rate(y_true: str, y_pred: str) -> Expr:
    return (pl.col(y_true).not_() & pl.col(y_pred)).sum() / pl.col(y_true).not_().sum()


def false_negative_rate(y_true: str, y_pred: str) -> Expr:
    return (pl.col(y_true) & pl.col(y_pred).not_()).sum() / pl.col(y_true).sum()


class AUC(LazyComponent):
    def __init__(self, x: str, y: str, *, name: str = "auc"):
        self.x = x
        self.y = y
        self.variable_name = name

    def execute2(self, data: LazyFrame) -> LazyFrame:
        return (
            data.select(self.x, self.y)
            .sort(self.x, maintain_order=True)
            .unique(maintain_order=True)
            .filter(is_internal_point(self.x, self.y).not_())
            .select(trapezoid(self.x, self.y).alias(self.variable_name))
        )


class ROCAUC(LazyComponent):
    def __init__(
        self, y_true: str, y_pred: ColumnNameOrSelector | Iterable[ColumnNameOrSelector]
    ):
        self.y_true = y_true
        self.y_pred = y_pred

    def execute2(self, data: LazyFrame) -> LazyFrame:
        n_pos = data.select(pl.col(self.y_true)).sum().collect().item()
        n_neg = data.select(pl.col(self.y_true).not_()).sum().collect().item()
        n_true_name = uuid.uuid4().hex
        n_false_name = uuid.uuid4().hex
        return pl.concat(
            [
                pl.concat(
                    [
                        pl.DataFrame(
                            [
                                pl.Series(y_pred, [float("inf")]),
                                pl.Series(n_false_name, [0], dtype=pl.UInt32),
                                pl.Series(n_true_name, [0], dtype=pl.UInt32),
                            ]
                        ).lazy(),
                        data.group_by(y_pred)
                        .agg(
                            pl.col(self.y_true).not_().sum().alias(n_false_name),
                            pl.col(self.y_true).sum().alias(n_true_name),
                        )
                        .sort(y_pred, descending=True)
                        .select(
                            y_pred,
                            pl.col(n_false_name).cum_sum(),
                            pl.col(n_true_name).cum_sum(),
                        ),
                    ]
                )
                .filter(is_internal_point(n_false_name, n_true_name).not_())
                .select(
                    (trapezoid(n_false_name, n_true_name) * 0.5 / n_pos / n_neg).alias(
                        y_pred
                    )
                )
                for y_pred in data.select(self.y_pred).collect_schema().names()
            ],
            how="horizontal",
        )
