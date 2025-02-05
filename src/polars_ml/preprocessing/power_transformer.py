import uuid
from typing import Literal, Mapping, Self, Sequence

import polars as pl
from polars import DataFrame, Expr
from scipy import stats

from polars_ml import Component


class PowerTransformer(Component):
    def __init__(
        self,
        *column: str,
        by: str | Sequence[str] | None = None,
        method: Literal["boxcox", "yeojohnson"] = "boxcox",
    ):
        self.columns = column
        self.by = [by] if isinstance(by, str) else list(by) if by is not None else None
        self.method = method

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        if self.by:
            self.maxlog = data.group_by(self.by).agg(
                pl.map_groups(
                    column,
                    lambda x: (
                        boxcox_maxlog if self.method == "boxcox" else yeojohnson_maxlog
                    )(x[0]),
                    return_dtype=pl.Float64,
                )
                for column in self.columns
            )
        else:
            self.maxlog = data.select(
                pl.map_groups(
                    column,
                    lambda x: (
                        boxcox_maxlog if self.method == "boxcox" else yeojohnson_maxlog
                    )(x[0]),
                    return_dtype=pl.Float64,
                )
                for column in self.columns
            )
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        columns = (
            data.lazy()
            .select(set(data.columns) & set(self.columns))
            .collect_schema()
            .names()
        )
        if self.by:
            return (
                data.join(
                    self.maxlog.select(*self.by, *columns),
                    on=self.by,
                    how="left",
                    suffix="_lmbda",
                )
                .with_columns(
                    boxcox(pl.col(col), pl.col(f"{col}_lmbda"))
                    if self.method == "boxcox"
                    else yeojohnson(pl.col(col), pl.col(f"{col}_lmbda"))
                    for col in columns
                )
                .drop(f"{col}_lmbda" for col in columns)
            )
        else:
            key = uuid.uuid4().hex
            return (
                data.with_columns(pl.lit(0).alias(key))
                .join(
                    self.maxlog.select(*columns, pl.lit(0).alias(key)),
                    on=key,
                    how="left",
                    suffix="_lmbda",
                )
                .with_columns(
                    boxcox(pl.col(col), pl.col(f"{col}_lmbda"))
                    if self.method == "boxcox"
                    else yeojohnson(pl.col(col), pl.col(f"{col}_lmbda"))
                    for col in columns
                )
                .drop(key)
                .drop(f"{col}_lmbda" for col in columns)
            )


def boxcox_maxlog(x: pl.Series) -> float:
    return float(stats.boxcox(x.drop_nulls().to_numpy())[1])


def yeojohnson_maxlog(x: pl.Series) -> float:
    return float(stats.yeojohnson(x.drop_nulls().to_numpy())[1])  # type: ignore


def boxcox(x: Expr, lmbda: Expr) -> Expr:
    return pl.when(x != 0).then((x**lmbda - 1) / lmbda).otherwise(x.log())


def yeojohnson(x: Expr, lmbda: Expr) -> Expr:
    return (
        pl.when((x >= 0) & (lmbda != 0))
        .then(((x + 1) ** lmbda - 1) / lmbda)
        .when((x >= 0) & (lmbda == 0))
        .then((x + 1).log())
        .when((x < 0) & (lmbda != 2))
        .then(-((-x + 1) ** (2 - lmbda) - 1) / (2 - lmbda))
        .otherwise(-(-x + 1).log())
    )
