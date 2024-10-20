import uuid
from typing import Iterable, override

import numpy as np
import polars as pl
from polars import Expr, LazyFrame
from polars._typing import IntoExpr

from polars_ml.component import LazyComponent


def is_internal_point(x: str, y: str) -> Expr:
    return (
        (pl.col(x).shift(1) - pl.col(x)) * (pl.col(y).shift(-1) - pl.col(y))
        == (pl.col(x).shift(-1) - pl.col(x)) * (pl.col(y).shift(1) - pl.col(y))
    ).fill_null(False)


def trapezoid(x: str, y: str) -> Expr:
    return (
        (pl.col(x) - pl.col(x).shift(1)) * (pl.col(y) + pl.col(y).shift(1))
    ).sum() * 0.5


def auc(data: LazyFrame, x: str, y: str) -> LazyFrame:
    first_or_last_name = uuid.uuid4().hex
    return (
        data.select(x, y)
        .sort(x, maintain_order=True)
        .with_columns(
            ((pl.col(x) != pl.col(x).shift(-1)) | (pl.col(x) != pl.col(x).shift(1)))
            .fill_null(True)
            .alias(first_or_last_name),
        )
        .filter(pl.col(first_or_last_name))
        .drop(first_or_last_name)
        .unique([x, y], maintain_order=True)
        .filter(is_internal_point(x, y).not_())
        .select(trapezoid(x, y))
    )


def roc_curve(
    data: LazyFrame,
    pred: str,
    label: str,
    *,
    threshold_name: str = "threshold",
    fpr_name: str = "fpr",
    tpr_name: str = "tpr",
) -> LazyFrame:
    data = data.cast({label: pl.Boolean})
    n_pos = data.select(pl.col(label)).sum().collect().item()
    n_neg = data.select(pl.col(label).not_()).sum().collect().item()
    return pl.concat(
        [
            pl.DataFrame(
                [
                    pl.Series(fpr_name, [0.0]),
                    pl.Series(tpr_name, [0.0]),
                    pl.Series(threshold_name, [float("inf")]),
                ]
            ).lazy(),
            data.group_by(pred)
            .agg(
                pl.col(label).not_().sum().alias(fpr_name),
                pl.col(label).sum().alias(tpr_name),
            )
            .sort(pred, descending=True)
            .select(
                pl.col(fpr_name).cum_sum() / n_neg,
                pl.col(tpr_name).cum_sum() / n_pos,
                pl.col(pred).cast(pl.Float64).alias(threshold_name),
            )
            .filter(is_internal_point(fpr_name, tpr_name).not_()),
        ]
    )


def roc_auc(data: LazyFrame, pred: str, label: str) -> LazyFrame:
    fpr_name = uuid.uuid4().hex
    tpr_name = uuid.uuid4().hex
    threshold_name = uuid.uuid4().hex
    return roc_curve(
        data,
        pred,
        label,
        threshold_name=threshold_name,
        fpr_name=fpr_name,
        tpr_name=tpr_name,
    ).select((trapezoid(fpr_name, tpr_name)).alias(pred))


def pr_curve(
    data: LazyFrame,
    pred: str,
    label: str,
    *,
    threshold_name: str = "threshold",
    precision_name: str = "precision",
    recall_name: str = "recall",
) -> LazyFrame:
    data = data.cast({label: pl.Boolean})
    n_pos = data.select(pl.col(label)).sum().collect().item()
    n_pos_pred_name = uuid.uuid4().hex
    return pl.concat(
        [
            pl.DataFrame(
                [
                    pl.Series(recall_name, [0.0]),
                    pl.Series(precision_name, [1.0]),
                    pl.Series(threshold_name, [float("inf")]),
                ]
            ).lazy(),
            data.group_by(pred)
            .agg(
                pl.col(label).sum().alias(recall_name),
                pl.col(label).sum().alias(precision_name),
                pl.len().alias(n_pos_pred_name),
            )
            .sort(pred, descending=True)
            .with_columns(pl.col(n_pos_pred_name).cum_sum())
            .select(
                pl.col(recall_name).cum_sum() / n_pos,
                pl.col(precision_name).cum_sum() / pl.col(n_pos_pred_name),
                pl.col(pred).cast(pl.Float64).alias(threshold_name),
            )
            .filter(is_internal_point(recall_name, precision_name).not_()),
        ]
    )


def pr_auc(data: LazyFrame, pred: str, label: str) -> LazyFrame:
    precision_name = uuid.uuid4().hex
    recall_name = uuid.uuid4().hex
    threshold_name = uuid.uuid4().hex
    return pr_curve(
        data,
        pred,
        label,
        threshold_name=threshold_name,
        precision_name=precision_name,
        recall_name=recall_name,
    ).select((trapezoid(recall_name, precision_name)).alias(pred))


class AUC(LazyComponent):
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
        data = data.select(self.preds, self.label)
        data = pl.concat(
            [auc(data, col, self.label) for col in data.collect_schema().names()],
            how="horizontal",
        )
        if self.include_name:
            data = data.select(pl.lit("auc").alias("method"), pl.all())

        return data


class ROCAUC(LazyComponent):
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
        data = data.select(self.preds, self.label)
        data = pl.concat(
            [roc_auc(data, col, self.label) for col in data.collect_schema().names()],
            how="horizontal",
        )
        if self.include_name:
            data = data.select(pl.lit("roc_auc").alias("method"), pl.all())

        return data


class PrecisionRecallAUC(LazyComponent):
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
        data = data.select(self.preds, self.label)
        data = pl.concat(
            [pr_auc(data, col, self.label) for col in data.collect_schema().names()],
            how="horizontal",
        )
        if self.include_name:
            data = data.select(pl.lit("pr_auc").alias("method"), pl.all())

        return data
