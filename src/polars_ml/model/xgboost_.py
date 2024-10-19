from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Self, override

import polars as pl
import xgboost as xgb
from polars import DataFrame
from polars._typing import IntoExpr
from tqdm import tqdm

from polars_ml.component import Component
from polars_ml.exception import NotFittedError


class XGBoost(Component):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        target: IntoExpr | Iterable[IntoExpr],
        params: Dict[str, Any],
        *,
        pred_name: str = "xgboost",
        model_file: str | None = None,
        is_valid_column: str | None = None,
        num_boost_round: int = 10,
        feval: Any | None = None,
        callbacks: List[Any] | None = None,
        plot_trees: bool = False,
    ):
        self.features = features
        self.target = target
        self.params = params or {}
        self.pred_name = pred_name
        self.is_valid_column = is_valid_column

        self.num_boost_round = num_boost_round
        self.feval = feval
        self.callbacks = callbacks or []

        self._plot_trees = plot_trees

        self.model: xgb.Booster | None = None

        if model_file:
            self.model = xgb.Booster()
            self.model.load_model(model_file)
            self._is_fitted = True

    @override
    def fit(self, data: DataFrame) -> Self:
        if self.is_valid_column in data.collect_schema().names():
            train_data = data.filter(pl.col(self.is_valid_column).not_()).drop(
                self.is_valid_column
            )
            train_features = train_data.select(self.features)
            train_target = train_data.select(self.target)
            dtrain = xgb.DMatrix(
                train_features.to_numpy(), label=train_target.to_numpy().squeeze()
            )

            valid_data = data.filter(pl.col(self.is_valid_column)).drop(
                self.is_valid_column
            )
            valid_features = valid_data.select(self.features)
            valid_target = valid_data.select(self.target)
            dvalid = xgb.DMatrix(
                valid_features.to_numpy(), label=valid_target.to_numpy().squeeze()
            )

            evals = [(dtrain, "train"), (dvalid, "valid")]
        else:
            train_features = data.select(self.features)
            train_target = data.select(self.target)
            dtrain = xgb.DMatrix(
                train_features.to_numpy(), label=train_target.to_numpy().squeeze()
            )
            evals = [(dtrain, "train")]

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            feval=self.feval,
            callbacks=self.callbacks,
        )
        self._is_fitted = True

        if log_dir := self.log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_model(str(log_dir / "model.json"))
            self.plot_feature_importance("weight", log_dir)
            self.plot_feature_importance("gain", log_dir)

            if self._plot_trees:
                trees_dir = log_dir / "trees"
                trees_dir.mkdir(parents=True, exist_ok=True)
                self.plot_trees(trees_dir)

        return self

    @override
    def execute(self, data: DataFrame) -> DataFrame:
        if self.model is None:
            raise NotFittedError()

        input = data.select(self.features).to_numpy()
        dmatrix = xgb.DMatrix(input)
        pred = self.model.predict(dmatrix)

        if pred.ndim == 2:
            schema = [f"{self.pred_name}_{i}" for i in range(pred.shape[1])]
        else:
            schema = [self.pred_name]

        output = pl.from_numpy(pred, schema=schema)
        return pl.concat([data, output], how="horizontal")

    def plot_feature_importance(
        self,
        importance_type: Literal[
            "weight", "gain", "cover", "total_gain", "total_cover"
        ],
        log_dir: Path,
    ):
        if self.model is None:
            raise NotFittedError()

        import matplotlib.pyplot as plt

        importance = self.model.get_score(importance_type=importance_type)
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.barh(list(importance.keys()), list(importance.values()))
        ax.set_title(f"Feature Importance ({importance_type})")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(log_dir / f"feature_importance_{importance_type}.png")
        fig.clear()
        plt.close(fig)

    def plot_trees(self, log_dir: Path):
        if self.model is None:
            raise NotFittedError()

        import matplotlib.pyplot as plt

        for i in tqdm(range(self.model.num_boosted_rounds())):
            fig, ax = plt.subplots(figsize=(20, 16))
            xgb.plot_tree(self.model, num_trees=i, ax=ax)
            fig.savefig(log_dir / f"{i}.png")
            fig.clear()
            plt.close(fig)
