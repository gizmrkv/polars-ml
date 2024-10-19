from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Self, override

import lightgbm as lgb
import numpy as np
import polars as pl
from polars import DataFrame
from polars._typing import IntoExpr
from tqdm import tqdm

from polars_ml.component import Component
from polars_ml.exception import NotFittedError


class LightGBM(Component):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        target: IntoExpr | Iterable[IntoExpr],
        params: Dict[str, Any],
        *,
        pred_name: str = "lightgbm",
        model_file: str | None = None,
        is_valid_column: str | None = None,
        num_boost_round: int = 10,
        feval: Any | None = None,
        callbacks: List[Any] | None = None,
        plot_trees: bool = False,
    ):
        import lightgbm as lgb

        self.features = features
        self.target = target
        self.params = params or {}
        self.pred_name = pred_name
        self.is_valid_column = is_valid_column

        self.num_boost_round = num_boost_round
        self.feval = feval
        self.callbacks = callbacks or []

        self._plot_trees = plot_trees

        self.booster: lgb.Booster | None = None

        if model_file:
            self.booster = lgb.Booster(model_file=model_file)
            self._is_fitted = True

    @override
    def fit(self, data: DataFrame) -> Self:
        import lightgbm as lgb

        if self.is_valid_column in data.collect_schema().names():
            train_data = data.filter(pl.col(self.is_valid_column).not_()).drop(
                self.is_valid_column
            )
            train_features = train_data.select(self.features)
            train_target = train_data.select(self.target)
            train_dataset = lgb.Dataset(
                train_features.to_numpy(),
                label=train_target.to_numpy().squeeze(),
                feature_name=train_features.columns,
                free_raw_data=False,
            )

            valid_data = data.filter(pl.col(self.is_valid_column)).drop(
                self.is_valid_column
            )
            valid_features = valid_data.select(self.features)
            valid_target = valid_data.select(self.target)
            valid_dataset = train_dataset.create_valid(
                valid_features.to_numpy(), valid_target.to_numpy().squeeze()
            )

            valid_sets = [train_dataset, valid_dataset]
            valid_names = ["train", "valid"]
        else:
            train_features = data.select(self.features)
            train_target = data.select(self.target)
            train_dataset = lgb.Dataset(
                train_features.to_numpy(),
                label=train_target.to_numpy().squeeze(),
                feature_name=train_features.columns,
                free_raw_data=False,
            )

            valid_sets = [train_dataset]
            valid_names = ["train"]

        self.booster = lgb.train(
            self.params,
            train_dataset,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=self.num_boost_round,
            feval=self.feval,
            callbacks=self.callbacks,
        )
        self._is_fitted = True

        if log_dir := self.log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.booster.save_model(log_dir / "model.txt")
            self.plot_feature_importance("split", log_dir)
            self.plot_feature_importance("gain", log_dir)

            if self._plot_trees:
                trees_dir = log_dir / "trees"
                trees_dir.mkdir(parents=True, exist_ok=True)
                self.plot_trees(trees_dir)

        return self

    @override
    def execute(self, data: DataFrame) -> DataFrame:
        if self.booster is None:
            raise NotFittedError()

        input = data.select(self.features).to_numpy()
        pred: np.ndarray[Any, Any] = self.booster.predict(input)  # type: ignore

        if pred.ndim == 2:
            schema = [f"{self.pred_name}_{i}" for i in range(pred.shape[1])]
        else:
            schema = [self.pred_name]

        output = pl.from_numpy(pred, schema=schema)
        return pl.concat([data, output], how="horizontal")

    def plot_feature_importance(
        self, importance_type: Literal["split", "gain"], log_dir: Path
    ):
        if self.booster is None:
            raise NotFittedError()

        import lightgbm as lgb
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 12))  # type: ignore
        lgb.plot_importance(
            self.booster, importance_type=importance_type, max_num_features=20, ax=ax
        )  # type: ignore
        fig.tight_layout()
        fig.savefig(log_dir / f"feature_importance_{importance_type}.png")  # type: ignore
        fig.clear()
        plt.close(fig)

    def plot_trees(self, log_dir: Path):
        if self.booster is None:
            raise NotFittedError()

        import lightgbm as lgb
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(20, 16))  # type: ignore
        for i in tqdm(range(self.booster.num_trees())):
            lgb.plot_tree(self.booster, ax=ax, tree_index=i)  # type: ignore
            fig.savefig(log_dir / f"{i}.png")  # type: ignore
            fig.clear()
            plt.close(fig)


class LGBMMetricsLogger:
    def __init__(self):
        self.num_iterations = 0
        self.metrics: Dict[str, Dict[str, List[float]]] = {}
        self.is_higher_better: Dict[str, bool] = {}

    def callback(self, env: "lgb.callback.CallbackEnv"):
        self.num_iterations = env.iteration
        if results := env.evaluation_result_list:
            for result in results:
                valid_name, eval_name, eval_value, is_higher_better = result  # type: ignore
                (
                    self.metrics.setdefault(valid_name, {})
                    .setdefault(eval_name, [])
                    .append(eval_value)
                )
                self.is_higher_better[eval_name] = is_higher_better
