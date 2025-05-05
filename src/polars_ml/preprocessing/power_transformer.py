import uuid
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Sequence

import polars as pl
from polars import DataFrame, Expr, Series
from polars._typing import ColumnNameOrSelector
from scipy import stats

from polars_ml.component import Component

if TYPE_CHECKING:
    from polars_ml import Pipeline


class PowerTransformer(Component):
    def __init__(
        self,
        *columns: str,
        by: str | Sequence[str] | None = None,
        method: Literal["boxcox", "yeojohnson"] = "boxcox",
    ):
        self.columns = columns
        self.by = [by] if isinstance(by, str) else list(by) if by is not None else []
        self.method = method
        self.suffix = uuid.uuid4().hex

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> "PowerTransformer":
        data = data.select(*self.columns, *self.by)
        self.maxlog_columns = data.columns
        exprs = [
            pl.col(column).map_batches(
                boxcox_maxlog if self.method == "boxcox" else yeojohnson_maxlog,
                return_dtype=pl.Float64,
                returns_scalar=True,
            )
            for column in self.maxlog_columns
        ]
        if self.by:
            self.maxlog = data.group_by(self.by).agg(*exprs)
        else:
            self.maxlog = data.select(*exprs)

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        targets = set(data.columns) & set(self.maxlog_columns)
        on_args: dict[str, Any] = (
            {"on": self.by}
            if self.by
            else {"left_on": pl.lit(0), "right_on": pl.lit(0)}
        )
        return (
            data.join(
                self.maxlog.select(*self.by, *targets),
                how="left",
                **on_args,
                suffix=self.suffix,
            )
            .with_columns(
                boxcox(pl.col(t), pl.col(f"{t}{self.suffix}"))
                if self.method == "boxcox"
                else yeojohnson(pl.col(t), pl.col(f"{t}{self.suffix}"))
                for t in targets
            )
            .select(data.columns)
        )

    def inverser(
        self, mapping: Mapping[str, str] | None = None
    ) -> "InversePowerTransformer":
        return InversePowerTransformer(self, mapping)


class InversePowerTransformer(Component):
    def __init__(
        self,
        power_transformer: PowerTransformer,
        inverse_mapping: Mapping[str, str] | None = None,
    ):
        self.power_transformer = power_transformer
        self.inverse_mapping = inverse_mapping

    def transform(self, data: DataFrame) -> DataFrame:
        mapping = self.inverse_mapping or {
            col: col for col in self.power_transformer.maxlog_columns
        }
        targets = set(data.columns) & set(mapping.keys())
        sources = set(data.columns) & set(mapping.values())
        on_args: dict[str, Any] = (
            {"on": self.power_transformer.by}
            if self.power_transformer.by
            else {"left_on": pl.lit(0), "right_on": pl.lit(0)}
        )
        return (
            data.join(
                self.power_transformer.maxlog.select(
                    *self.power_transformer.by, *targets, *sources
                ),
                how="left",
                **on_args,
                suffix=self.power_transformer.suffix,
            )
            .with_columns(
                boxcox_inv(pl.col(t), pl.col(f"{s}{self.power_transformer.suffix}"))
                if self.power_transformer.method == "boxcox"
                else yeojohnson_inv(
                    pl.col(t), pl.col(f"{s}{self.power_transformer.suffix}")
                )
                for t, s in {t: s for t, s in mapping.items() if t in targets}.items()
            )
            .select(data.columns)
        )


def boxcox_maxlog(x: Series) -> float:
    return float(stats.boxcox(x.drop_nulls().to_numpy())[1])


def boxcox(x: Expr, lmbda: Expr) -> Expr:
    return pl.when(lmbda != 0).then((x**lmbda - 1) / lmbda).otherwise(x.log())


def boxcox_inv(x: Expr, lmbda: Expr) -> Expr:
    return pl.when(lmbda != 0).then((x * lmbda + 1) ** (1 / lmbda)).otherwise(x.exp())


def yeojohnson_maxlog(x: Series) -> float:
    return float(stats.yeojohnson(x.drop_nulls().to_numpy())[1])  # type: ignore


def yeojohnson(x: Expr, lmbda: Expr) -> Expr:
    return (
        pl.when((x >= 0) & (lmbda != 0))
        .then(((x + 1) ** lmbda - 1) / lmbda)
        .when((x >= 0) & (lmbda == 0))
        .then((x + 1).log())
        .when((x < 0) & (lmbda != 2))
        .then(((-x + 1) ** (2 - lmbda) - 1) / (lmbda - 2))
        .otherwise(-(-x + 1).log())
    )


def yeojohnson_inv(x: Expr, lmbda: Expr) -> Expr:
    return (
        pl.when((x >= 0) & (lmbda != 0))
        .then((x * lmbda + 1) ** (1 / lmbda) - 1)
        .when((x >= 0) & (lmbda == 0))
        .then(x.exp() - 1)
        .when((x < 0) & (lmbda != 2))
        .then(1 - (x * lmbda - 2 * x + 1) ** (1 / (2 - lmbda)))
        .otherwise(1 - (x.exp() - 1))
    )
