from __future__ import annotations

import itertools
from typing import Iterable, Literal, Self

import polars as pl
from polars import DataFrame, Expr
from polars._typing import ColumnNameOrSelector, CorrelationMethod
from tqdm import tqdm

from polars_ml.base import Transformer


class ArithmeticSynthesis(Transformer):
    def __init__(
        self,
        columns: ColumnNameOrSelector | Iterable[ColumnNameOrSelector],
        *,
        order: int,
        method: Literal["additive", "multiplicative"] = "additive",
        drop_high_correlation_features_method: CorrelationMethod | None = None,
        threshold: float = 0.9,
        show_progress: bool = True,
    ):
        self.selector = columns
        self.order = order
        self.method = method
        self.drop_high_correlation_features_method = (
            drop_high_correlation_features_method
        )
        self.threshold = threshold
        self.show_progress = show_progress

        assert order >= 1, "Order must be at least 1"

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        self.columns = data.select(self.selector).columns
        self.exprs: dict[str, Expr] = {}
        columns = data.select(self.columns).collect_schema().names()
        for comb in tqdm(
            itertools.combinations(columns, self.order + 1),
            disable=not self.show_progress,
        ):
            n = len(comb)
            anchor, *comb = comb
            for r in tqdm(
                range(0, n // 2 + 1), disable=not self.show_progress, leave=False
            ):
                for denos in itertools.combinations(comb, r):
                    numes = [anchor] + [c for c in comb if c not in denos]
                    nume, *more_numes = numes
                    expr = pl.col(nume)

                    if self.method == "additive":
                        for c in more_numes:
                            expr = expr + pl.col(c)
                        for d in denos:
                            expr = expr - pl.col(d)

                        name = "+".join(numes)
                        if len(denos) > 0:
                            name += "-" + "-".join(denos)
                    else:
                        for c in more_numes:
                            expr = expr * pl.col(c)
                        for d in denos:
                            expr = expr / pl.col(d)

                        name = "*".join(numes)
                        if len(denos) > 0:
                            name += "/" + "/".join(denos)

                    self.exprs[name] = expr.alias(name)

        if self.show_progress:
            print(f"Generated {len(self.exprs)} features")

        if self.drop_high_correlation_features_method is not None:
            (n, e), *exprs_list = self.exprs.items()
            new_exprs = {n: e}
            while True:
                (n, e), *exprs_list = exprs_list
                new_exprs[n] = e

                if len(exprs_list) == 0:
                    break

                low_corr_columns = set(
                    data.select(
                        pl.corr(
                            e2,
                            e,
                            method=self.drop_high_correlation_features_method,  # type: ignore
                        )
                        for _, e2 in exprs_list
                    )
                    .unpivot()
                    .filter(pl.col("value").abs() <= self.threshold)["variable"]
                    .to_list()
                )

                exprs_list = [(n, e) for (n, e) in exprs_list if n in low_corr_columns]

            self.exprs = new_exprs

            if self.show_progress:
                print(
                    f"Kept {len(self.exprs)} features after dropping high correlation features"
                )

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        return data.with_columns(self.exprs.values())
