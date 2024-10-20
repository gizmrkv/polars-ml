from typing import Iterable, override

import polars as pl
from polars import Expr, LazyFrame
from polars._typing import IntoExpr

from polars_ml.component import LazyComponent


def precision(preds: Expr, label: Expr) -> Expr:
    return (label & preds).sum() / preds.sum()


def recall(preds: Expr, label: Expr) -> Expr:
    return (label & preds).sum() / label.sum()


def f_score(preds: Expr, label: Expr, *, beta: float = 1.0) -> Expr:
    p = precision(preds, label)
    r = recall(preds, label)
    return (1.0 + beta**2.0) * p * r / (beta**2.0 * p + r)


def true_negative_rate(preds: Expr, label: Expr) -> Expr:
    return (label.not_() & preds.not_()).sum() / label.not_().sum()


def false_negative_rate(preds: Expr, label: Expr) -> Expr:
    return (label & preds.not_()).sum() / label.sum()


def false_positive_rate(preds: Expr, label: Expr) -> Expr:
    return (label.not_() & preds).sum() / label.not_().sum()


def true_positive_rate(preds: Expr, label: Expr) -> Expr:
    return (label & preds).sum() / label.sum()


def log_loss(preds: Expr, label: Expr) -> Expr:
    return -(label * preds.log() + (1 - label) * (1 - preds).log()).mean()


class Precision(LazyComponent):
    def __init__(
        self,
        preds: IntoExpr | Iterable[IntoExpr],
        label: str,
        *,
        include_name: bool = False,
    ):
        self.preds = preds
        self.label = label
        self.include_name = include_name
        self._is_fitted = True

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        data = data.select(self.preds, self.label).select(
            precision(pl.all().exclude(self.label), pl.col(self.label))
        )
        if self.include_name:
            data = data.select(pl.lit("precision").alias("method"), pl.all())

        return data


class Recall(LazyComponent):
    def __init__(
        self,
        preds: IntoExpr | Iterable[IntoExpr],
        label: str,
        *,
        include_name: bool = False,
    ):
        self.preds = preds
        self.label = label
        self.include_name = include_name
        self._is_fitted = True

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        data = data.select(self.preds, self.label).select(
            recall(pl.all().exclude(self.label), pl.col(self.label))
        )
        if self.include_name:
            data = data.select(pl.lit("recall").alias("method"), pl.all())

        return data


class FScore(LazyComponent):
    def __init__(
        self,
        preds: IntoExpr | Iterable[IntoExpr],
        label: str,
        *,
        beta: float = 1.0,
        include_name: bool = False,
    ):
        self.preds = preds
        self.label = label
        self.beta = beta
        self.include_name = include_name
        self._is_fitted = True

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        data = data.select(self.preds, self.label).select(
            f_score(pl.all().exclude(self.label), pl.col(self.label), beta=self.beta)
        )
        if self.include_name:
            data = data.select(pl.lit(f"f-{self.beta}").alias("method"), pl.all())

        return data


class TrueNegativeRate(LazyComponent):
    def __init__(
        self,
        preds: IntoExpr | Iterable[IntoExpr],
        label: str,
        *,
        include_name: bool = False,
    ):
        self.preds = preds
        self.label = label
        self.include_name = include_name
        self._is_fitted = True

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        data = data.select(self.preds, self.label).select(
            true_negative_rate(pl.all().exclude(self.label), pl.col(self.label))
        )
        if self.include_name:
            data = data.select(pl.lit("tnr").alias("method"), pl.all())

        return data


class FalseNegativeRate(LazyComponent):
    def __init__(
        self,
        preds: IntoExpr | Iterable[IntoExpr],
        label: str,
        *,
        include_name: bool = False,
    ):
        self.preds = preds
        self.label = label
        self.include_name = include_name
        self._is_fitted = True

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        data = data.select(self.preds, self.label).select(
            false_negative_rate(pl.all().exclude(self.label), pl.col(self.label))
        )
        if self.include_name:
            data = data.select(pl.lit("fnr").alias("method"), pl.all())

        return data


class FalsePositiveRate(LazyComponent):
    def __init__(
        self,
        preds: IntoExpr | Iterable[IntoExpr],
        label: str,
        *,
        include_name: bool = False,
    ):
        self.preds = preds
        self.label = label
        self.include_name = include_name
        self._is_fitted = True

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        data = data.select(self.preds, self.label).select(
            false_positive_rate(pl.all().exclude(self.label), pl.col(self.label))
        )
        if self.include_name:
            data = data.select(pl.lit("fpr").alias("method"), pl.all())

        return data


class TruePositiveRate(LazyComponent):
    def __init__(
        self,
        preds: IntoExpr | Iterable[IntoExpr],
        label: str,
        *,
        include_name: bool = False,
    ):
        self.preds = preds
        self.label = label
        self.include_name = include_name
        self._is_fitted = True

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        data = data.select(self.preds, self.label).select(
            true_positive_rate(pl.all().exclude(self.label), pl.col(self.label))
        )
        if self.include_name:
            data = data.select(pl.lit("tpr").alias("method"), pl.all())

        return data


class LogLoss(LazyComponent):
    def __init__(
        self,
        preds: IntoExpr | Iterable[IntoExpr],
        label: str,
        *,
        include_name: bool = False,
    ):
        self.preds = preds
        self.label = label
        self.include_name = include_name
        self._is_fitted = True

    @override
    def execute(self, data: LazyFrame) -> LazyFrame:
        data = data.select(self.preds, self.label).select(
            log_loss(pl.all().exclude(self.label), pl.col(self.label))
        )
        if self.include_name:
            data = data.select(pl.lit("log_loss").alias("method"), pl.all())

        return data
