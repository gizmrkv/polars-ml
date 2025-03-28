import uuid
from typing import TYPE_CHECKING, Any, Iterable, Literal, Mapping, Self, Sequence

import polars as pl
from polars import DataFrame, Expr
from polars._typing import ColumnNameOrSelector

from polars_ml.pipeline.component import PipelineComponent

if TYPE_CHECKING:
    from polars_ml import Pipeline


class Scaler(PipelineComponent):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *more_columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        by: str | Sequence[str] | None = None,
        method: Literal["standard", "min-max", "robust"] = "standard",
        quantile: tuple[float, float] = (0.25, 0.75),
    ):
        self.columns = columns
        self.more_columns = more_columns
        self.by = [by] if isinstance(by, str) else list(by) if by is not None else []
        self.method = method
        self.q1, self.q3 = quantile
        self.suffix = uuid.uuid4().hex

        assert 0 <= self.q1 < self.q3 <= 1, (
            f"Quantile values must be in the range [0, 1] and q1 < q3. Got {quantile}"
        )

    def move_expr(self, column: str) -> Expr:
        if self.method == "standard":
            return pl.col(column).mean()
        elif self.method == "min-max":
            return pl.col(column).min()
        elif self.method == "robust":
            return pl.col(column).median()
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def scale_expr(self, column: str) -> Expr:
        if self.method == "standard":
            return pl.col(column).std()
        elif self.method == "min-max":
            return pl.col(column).max() - pl.col(column).min()
        elif self.method == "robust":
            return pl.col(column).quantile(self.q3) - pl.col(column).quantile(self.q1)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        data = data.select(self.columns, *self.more_columns, *self.by)

        self.move_scale_columns = [c for c in data.columns if c not in self.by]
        exprs = [
            self.move_expr(col).alias(f"{col}_move_{self.suffix}")
            for col in self.move_scale_columns
        ] + [
            self.scale_expr(col).alias(f"{col}_scale_{self.suffix}")
            for col in self.move_scale_columns
        ]
        if self.by:
            self.move_scale = data.group_by(self.by).agg(*exprs)
        else:
            self.move_scale = data.select(*exprs)

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        targets = set(data.columns) & set(self.move_scale_columns)
        on_args: dict[str, Any] = (
            {"on": self.by}
            if self.by
            else {"left_on": pl.lit(0), "right_on": pl.lit(0)}
        )
        return (
            data.join(
                self.move_scale.select(
                    *self.by,
                    *[f"{t}_move_{self.suffix}" for t in targets],
                    *[f"{t}_scale_{self.suffix}" for t in targets],
                ),
                how="left",
                **on_args,
            )
            .with_columns(
                (pl.col(t) - pl.col(f"{t}_move_{self.suffix}"))
                / pl.col(f"{t}_scale_{self.suffix}")
                for t in targets
            )
            .select(data.columns)
        )


class ScalerInverse(PipelineComponent):
    def __init__(
        self, scaler: Scaler, inverse_mapping: Mapping[str, str] | None = None
    ):
        self.scaler = scaler
        self.inverse_mapping = inverse_mapping

    def transform(self, data: DataFrame) -> DataFrame:
        mapping = self.inverse_mapping or {
            col: col for col in self.scaler.move_scale_columns
        }
        targets = set(data.columns) & set(mapping.keys())
        sources = set(data.columns) & set(mapping.values())
        on_args: dict[str, Any] = (
            {"on": self.scaler.by}
            if self.scaler.by
            else {"left_on": pl.lit(0), "right_on": pl.lit(0)}
        )
        suffix = self.scaler.suffix
        return (
            data.join(
                self.scaler.move_scale.select(
                    *self.scaler.by,
                    *[f"{col}_move_{suffix}" for col in sources],
                    *[f"{col}_scale_{suffix}" for col in sources],
                ),
                how="left",
                **on_args,
            )
            .with_columns(
                (pl.col(t) * pl.col(f"{s}_scale_{suffix}"))
                + pl.col(f"{s}_move_{suffix}")
                for t, s in {t: s for t, s in mapping.items() if t in targets}.items()
            )
            .select(data.columns)
        )


class ScalerInverseContext:
    def __init__(
        self,
        pipeline: "Pipeline",
        scaler: Scaler,
        mapping: Mapping[str, str] | None = None,
        *,
        component_name: str | None = None,
    ):
        self.pipeline = pipeline
        self.scaler = scaler
        self.mapping = mapping
        self.component_name = component_name

    def __enter__(self) -> "Pipeline":
        self.pipeline.pipe(self.scaler, component_name=self.component_name)
        return self.pipeline

    def __exit__(self, *args: Any, **kwargs: Any):
        self.pipeline.pipe(
            ScalerInverse(self.scaler, inverse_mapping=self.mapping),
            component_name=self.component_name + "_inverse"
            if self.component_name
            else None,
        )
