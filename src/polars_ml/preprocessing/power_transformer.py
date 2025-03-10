import uuid
from typing import TYPE_CHECKING, Any, Literal, Mapping, Self, Sequence

import polars as pl
from polars import DataFrame, Expr
from scipy import stats

from polars_ml.pipeline.component import PipelineComponent

if TYPE_CHECKING:
    from polars_ml import Pipeline


class PowerTransformer(PipelineComponent):
    def __init__(
        self,
        *column: str,
        by: str | Sequence[str] | None = None,
        method: Literal["boxcox", "yeojohnson"] = "boxcox",
    ):
        self.columns = column
        self.by = [by] if isinstance(by, str) else list(by) if by is not None else None
        self.method = method
        self.suffix = uuid.uuid4().hex

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        exprs = [
            pl.map_groups(
                column,
                lambda x: (
                    boxcox_maxlog if self.method == "boxcox" else yeojohnson_maxlog
                )(x[0]),
                return_dtype=pl.Float64,
            )
            for column in self.columns
        ]
        if self.by:
            self.maxlog = data.group_by(self.by).agg(*exprs)
        else:
            self.maxlog = data.select(*exprs)

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        targets = set(data.columns) & set(self.columns)
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


class PowerTransformerInverse(PipelineComponent):
    def __init__(
        self,
        power_transformer: PowerTransformer,
        mapping: Mapping[str, str] | None = None,
    ):
        self.power_transformer = power_transformer
        self.mapping = mapping

    def transform(self, data: DataFrame) -> DataFrame:
        mapping = self.mapping or {col: col for col in self.power_transformer.columns}
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


class PowerTransformerInverseContext:
    def __init__(
        self,
        pipeline: "Pipeline",
        power_transformer: PowerTransformer,
        mapping: Mapping[str, str] | None = None,
        *,
        component_name: str | None = None,
    ):
        self.pipeline = pipeline
        self.power_transformer = power_transformer
        self.mapping = mapping
        self.component_name = component_name

    def __enter__(self) -> "Pipeline":
        self.pipeline.pipe(self.power_transformer, component_name=self.component_name)
        return self.pipeline

    def __exit__(self, *args: Any, **kwargs: Any):
        self.pipeline.pipe(
            PowerTransformerInverse(self.power_transformer, mapping=self.mapping),
            component_name=self.component_name + "_inverse"
            if self.component_name
            else None,
        )


def boxcox_maxlog(x: pl.Series) -> float:
    return float(stats.boxcox(x.drop_nulls().to_numpy())[1])


def boxcox(x: Expr, lmbda: Expr) -> Expr:
    return pl.when(lmbda != 0).then((x**lmbda - 1) / lmbda).otherwise(x.log())


def boxcox_inv(x: Expr, lmbda: Expr) -> Expr:
    return pl.when(lmbda != 0).then((x * lmbda + 1) ** (1 / lmbda)).otherwise(x.exp())


def yeojohnson_maxlog(x: pl.Series) -> float:
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
