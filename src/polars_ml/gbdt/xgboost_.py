from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Mapping,
    Self,
)

import numpy as np
import polars as pl
import polars.selectors as cs
from numpy.typing import NDArray
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.base import Transformer

if TYPE_CHECKING:
    import xgboost as xgb


class XGBoost(Transformer):
    def __init__(
        self,
        params: Mapping[str, Any],
        label: IntoExpr,
        features: IntoExpr | Iterable[IntoExpr] | None = None,
        *,
        prediction_name: str = "prediction",
        out_dir: str | Path | None = None,
    ):
        self.params = params
        self.label = label
        self.features_selector = features
        self.prediction_name = prediction_name
        self.out_dir = Path(out_dir) if out_dir else None

    def get_booster(self) -> xgb.Booster:
        """Return the trained booster."""
        return self.booster

    def create_train(self, data: DataFrame) -> xgb.DMatrix:
        """Create an XGBoost DMatrix for training."""
        import xgboost as xgb

        if self.features_selector is None:
            # Determine features lazily if not specified
            label_cols = data.lazy().select(self.label).collect_schema().names()
            self.features_selector = cs.exclude(*label_cols)

        features = data.select(self.features_selector)
        label = data.select(self.label)
        self.feature_names = features.columns

        return xgb.DMatrix(
            features.to_pandas(),
            label=label.to_pandas(),
            feature_names=self.feature_names,
        )

    def create_valid(self, data: DataFrame, reference: xgb.DMatrix) -> xgb.DMatrix:
        """Create an XGBoost DMatrix for validation."""
        import xgboost as xgb

        # Use stored feature_names to ensure consistency
        features = data.select(self.feature_names)
        label = data.select(self.label)

        return xgb.DMatrix(
            features.to_pandas(),
            label=label.to_pandas(),
            feature_names=self.feature_names,
        )

    def make_train_valid_sets(
        self, data: DataFrame, **more_data: DataFrame
    ) -> tuple[xgb.DMatrix, list[tuple[xgb.DMatrix, str]]]:
        """Prepare training and validation DMatrices."""
        dtrain = self.create_train(data)
        evals = []
        for name, valid_data in more_data.items():
            dvalid = self.create_valid(valid_data, dtrain)
            evals.append((dvalid, name))

        evals.append((dtrain, "train"))
        return dtrain, evals

    def predict(self, data: DataFrame) -> NDArray:
        """Generate raw predictions using the booster."""
        import xgboost as xgb

        # Use stored feature_names to ensure consistency with training
        input_data = xgb.DMatrix(
            data.select(self.feature_names).to_pandas(),
            feature_names=self.feature_names,
        )
        booster = self.get_booster()

        return booster.predict(input_data)

    def transform(self, data: DataFrame) -> DataFrame:
        """Transform the data by adding prediction columns."""
        pred = self.predict(data)
        name = self.prediction_name

        prediction_df = pl.from_numpy(
            pred,
            schema=[name]
            if pred.ndim == 1
            else [f"{name}_{i}" for i in range(pred.shape[1])],
        )

        return pl.concat([data, prediction_df], how="horizontal")

    def save(self, out_dir: str | Path) -> None:
        """Save the booster and relevant metadata."""
        booster = self.get_booster()
        out_dir = Path(out_dir)
        save_xgboost_booster(booster, out_dir)

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        import xgboost as xgb

        dtrain, evals = self.make_train_valid_sets(data, **more_data)

        self.booster = xgb.train(
            self.params,
            dtrain,
            evals=evals,
        )

        if self.out_dir:
            self.save(self.out_dir)

        return self


def save_xgboost_booster(booster: xgb.Booster, out_dir: str | Path):
    import json

    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(out_dir / "model.json")

    config = json.loads(booster.save_config())
    with open(out_dir / "params.json", "w") as f:
        json.dump(config, f, indent=4)

    if feature_names := booster.feature_names:
        pl.DataFrame(
            {
                "feature": feature_names,
                **{
                    importance_type: [
                        booster.get_score(importance_type=importance_type).get(fn, 0.0)
                        for fn in feature_names
                    ]
                    for importance_type in [
                        "gain",
                        "weight",
                        "cover",
                        "total_gain",
                        "total_cover",
                    ]
                },
            }
        ).write_csv(out_dir / "feature_importance.csv")

    import xgboost as xgb

    xgb.plot_importance(booster, importance_type="gain")
    plt.savefig(out_dir / "importance_gain.png")
    plt.tight_layout()
    plt.close()

    xgb.plot_importance(booster, importance_type="weight")
    plt.savefig(out_dir / "importance_weight.png")
    plt.tight_layout()
    plt.close()
