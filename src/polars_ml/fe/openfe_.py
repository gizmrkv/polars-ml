import itertools
from typing import Any, Callable, Literal, Mapping, Self, Sequence

import lightgbm as lgb
import polars as pl
from numpy.typing import NDArray
from polars import DataFrame
from tqdm import tqdm

from polars_ml import Component
from polars_ml.fe import operator as op
from polars_ml.utils import deduplicate_scores, incremental_sampling


class OpenFE(Component):
    def __init__(
        self,
        label: str,
        *,
        init_score: str | Sequence[str],
        params_stage_1: dict[str, Any],
        metric_fn: Callable[[NDArray[Any], NDArray[Any]], float],
        direction: Literal["maximize", "minimize"] = "maximize",
        halving_ratio: float = 0.5,
        min_candidates: int = 2000,
        params_stage_2: dict[str, Any],
        n_best_features: int = 100,
        max_order: int = 1,
        numerical_features: Sequence[str],
        categorical_features: Sequence[str],
    ):
        self.label = label
        self.init_score = init_score
        self.params_stage_1 = params_stage_1
        self.metric_fn = metric_fn
        self.direction: Literal["maximize", "minimize"] = direction
        self.halving_ratio = halving_ratio
        self.min_candidates = min_candidates
        self.params_stage_2 = params_stage_2
        self.n_best_features = n_best_features
        self.max_order = max_order
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        self.new_features: list[op.Operator] = []

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        assert isinstance(validation_data, DataFrame), (
            "validation_data must be DataFrame"
        )

        new_features = self.enumerate_new_features()

        train_init_score = data.select(self.init_score).to_numpy().squeeze()
        valid_init_score = validation_data.select(self.init_score).to_numpy().squeeze()
        data = data.select(pl.exclude(self.init_score))
        validation_data = validation_data.select(pl.exclude(self.init_score))

        new_features_metric = self.screen_stage_1(
            data,
            validation_data,
            self.label,
            new_features,
            train_init_score=train_init_score,
            valid_init_score=valid_init_score,
        )
        new_features_metric = self.screen_stage_2(
            data,
            validation_data,
            self.label,
            new_features,
            params=self.params_stage_2,
            topk=self.n_best_features,
        )
        self.new_features = list(new_features_metric.keys())
        for f in self.new_features:
            f.fit(data)

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        return pl.concat(
            [
                data,
                *[
                    f.transform(data).select(pl.col(f.name).alias(str(f)))
                    for f in self.new_features
                ],
            ],
            how="horizontal",
        )

    def enumerate_new_features(self) -> list[op.Operator]:
        num_features: list[list[op.Operator]] = [
            [op.Column(c) for c in self.numerical_features]
        ]
        cat_features: list[list[op.Operator]] = [
            [op.Column(c) for c in self.categorical_features]
        ]

        for order in range(self.max_order):
            num_features.append([])
            cat_features.append([])

            for col in num_features[order]:
                num_features[-1].extend(
                    [
                        op.Minus(col),
                        op.Inv(col),
                        op.Pow(col),
                        op.Log(col),
                        op.Sqrt(col),
                    ]
                )

            for lhs_order in range(order + 1):
                rhs_order = order - lhs_order
                if lhs_order == rhs_order:
                    for lhs, rhs in itertools.combinations(num_features[lhs_order], 2):
                        num_features[-1].append(op.Add(lhs, rhs))
                        num_features[-1].append(op.Sub(lhs, rhs))
                        num_features[-1].append(op.Mul(lhs, rhs))
                        num_features[-1].append(op.Div(lhs, rhs))
                else:
                    for lhs, rhs in itertools.product(
                        num_features[lhs_order], num_features[rhs_order]
                    ):
                        num_features[-1].append(op.Add(lhs, rhs))
                        num_features[-1].append(op.Sub(lhs, rhs))
                        num_features[-1].append(op.Mul(lhs, rhs))
                        num_features[-1].append(op.Div(lhs, rhs))

                for num, cat in itertools.product(
                    num_features[lhs_order], cat_features[rhs_order]
                ):
                    num_features[-1].append(op.GroupByMean(cat, num))
                    num_features[-1].append(op.GroupByStd(cat, num))

                for val, cat in itertools.product(
                    itertools.chain(num_features[lhs_order], cat_features[rhs_order]),
                    cat_features[rhs_order],
                ):
                    if val != cat:
                        num_features[-1].append(op.GroupByNUnique(cat, val))

        return list(itertools.chain(*num_features[1:], *cat_features[1:]))

    def screen_stage_1(
        self,
        train_data: DataFrame,
        valid_data: DataFrame,
        label: str,
        new_features: Sequence[op.Operator],
        *,
        train_init_score: NDArray[Any],
        valid_init_score: NDArray[Any],
    ) -> dict[op.Operator, float]:
        valid_y = valid_data[label].to_numpy()
        scores: dict[op.Operator, float] = {}
        for indexes in incremental_sampling(train_data.height, 4):
            train_sub_df = train_data.select(pl.all().gather(indexes))
            train_sub_y = train_sub_df[label].to_numpy()
            train_sub_init_score = train_init_score[indexes]

            scores = {}
            for new_feature in tqdm(new_features):
                new_train_X = new_feature.fit_transform(train_sub_df).to_numpy()
                new_valid_X = new_feature.transform(valid_data).to_numpy()

                lgb_train = lgb.Dataset(
                    new_train_X, train_sub_y, init_score=train_sub_init_score
                )
                lgb_valid = lgb_train.create_valid(
                    new_valid_X, valid_y, init_score=valid_init_score
                )

                gbm = lgb.train(
                    self.params_stage_1,
                    lgb_train,
                    valid_sets=[lgb_valid],
                    num_boost_round=100,
                    callbacks=[lgb.early_stopping(3, verbose=False)],
                )

                pred_valid_y: NDArray[Any] = gbm.predict(new_valid_X)  # type: ignore
                curr_metric = self.metric_fn(valid_y, pred_valid_y)
                scores[new_feature] = curr_metric

            scores = deduplicate_scores(scores, direction=self.direction)
            next_features = list(scores.keys())
            new_features = next_features[
                : max(
                    int(len(next_features) * self.halving_ratio),
                    min(self.min_candidates, len(next_features)),
                )
            ]

        return scores

    def screen_stage_2(
        self,
        train_data: DataFrame,
        valid_data: DataFrame,
        label: str,
        new_features: Sequence[op.Operator],
        *,
        params: dict[str, Any],
        topk: int = 100,
    ) -> dict[op.Operator, float]:
        name2op = {f.name: f for f in new_features}
        new_train_data = pl.concat(
            [
                train_data,
                *[f.fit_transform(train_data) for f in new_features],
            ],
            how="horizontal",
        )
        new_valid_data = pl.concat(
            [
                valid_data,
                *[f.transform(valid_data) for f in new_features],
            ],
            how="horizontal",
        )

        train_X = new_train_data.drop(label).to_numpy()
        train_y = new_train_data[label].to_numpy()
        valid_X = new_valid_data.drop(label).to_numpy()
        valid_y = new_valid_data[label].to_numpy()

        lgb_train = lgb.Dataset(
            train_X, train_y, feature_name=new_train_data.drop(label).columns
        )
        lgb_valid = lgb_train.create_valid(valid_X, valid_y)

        gbm = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_valid],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10)],
        )

        importance = gbm.feature_importance(importance_type="gain")
        feature_names = gbm.feature_name()
        feature_importance = [
            (name, imp)
            for name, imp in zip(feature_names, importance)
            if name in name2op
        ]
        sorted_feature_importance = sorted(
            feature_importance, key=lambda x: x[1], reverse=True
        )
        return {name2op[name]: imp for name, imp in sorted_feature_importance[:topk]}
