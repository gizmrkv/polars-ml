from pathlib import Path
from typing import Any, Dict, Iterable, List, Self, override

import polars as pl
from catboost import CatBoost, Pool
from polars import DataFrame
from polars._typing import IntoExpr
from tqdm import tqdm

from polars_ml.component import Component
from polars_ml.exception import NotFittedError


class CatBoostModel(Component):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        target: IntoExpr | Iterable[IntoExpr],
        params: Dict[str, Any],
        *,
        pred_name: str = "catboost",
        model_file: str | None = None,
        is_valid_column: str | None = None,
        cat_features: List[str] | None = None,
        plot_trees: bool = False,
    ):
        self.features = features
        self.target = target
        self.params = params or {}
        self.pred_name = pred_name

        self._plot_trees = plot_trees
        self.is_valid_column = is_valid_column
        self.cat_features = cat_features

        self.model: CatBoost | None = None

        if model_file:
            self.model = CatBoost()
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
            train_pool = Pool(
                data=train_features.to_numpy(),
                label=train_target.to_numpy().squeeze(),
                feature_names=train_features.columns,
                cat_features=self.cat_features,
            )

            valid_data = data.filter(pl.col(self.is_valid_column)).drop(
                self.is_valid_column
            )
            valid_features = valid_data.select(self.features)
            valid_target = valid_data.select(self.target)
            valid_pool = Pool(
                data=valid_features.to_numpy(),
                label=valid_target.to_numpy().squeeze(),
                feature_names=valid_features.columns,
                cat_features=self.cat_features,
            )

            self.model = CatBoost(self.params)
            self.model.fit(train_pool, eval_set=valid_pool, plot=True)
        else:
            train_features = data.select(self.features)
            train_target = data.select(self.target)
            train_pool = Pool(
                data=train_features.to_numpy(),
                label=train_target.to_numpy().squeeze(),
                feature_names=train_features.columns,
                cat_features=self.cat_features,
            )

            self.model = CatBoost(self.params)
            self.model.fit(train_pool, plot=True)

        self._is_fitted = True

        if log_dir := self.log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_model(str(log_dir / "model.cbm"))
            self.plot_feature_importance(log_dir)

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
        pred = self.model.predict(input)

        if pred.ndim == 2:
            schema = [f"{self.pred_name}_{i}" for i in range(pred.shape[1])]
        else:
            schema = [self.pred_name]

        output = pl.from_numpy(pred, schema=schema)
        return pl.concat([data, output], how="horizontal")

    def plot_feature_importance(self, log_dir: Path):
        if self.model is None:
            raise NotFittedError()

        import matplotlib.pyplot as plt

        feature_importance = self.model.get_feature_importance()
        feature_names = self.model.feature_names_

        fig, ax = plt.subplots(figsize=(10, 12))
        ax.barh(feature_names, feature_importance)  # type: ignore
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(log_dir / "feature_importance.png")
        fig.clear()
        plt.close(fig)

    def plot_trees(self, log_dir: Path):
        if self.model is None:
            raise NotFittedError()

        import graphviz

        for i in tqdm(range(self.model.tree_count_)):  # type: ignore
            tree_graph: graphviz.Digraph = self.model.plot_tree(tree_idx=i, pool=None)
            tree_graph.render(
                filename=log_dir / f"tree_{i}", format="png", cleanup=True
            )
