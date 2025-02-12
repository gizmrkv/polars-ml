import itertools
import json
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Self, Sequence

import lightgbm as lgb
import polars as pl
from loguru import logger
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
        max_order: int = 1,
        numerical_features: Sequence[str],
        categorical_features: Sequence[str],
        n_subsamples: int = 8,
        params_stage_1: dict[str, Any],
        init_score: str | Sequence[str],
        metric_fn: Callable[[NDArray[Any], NDArray[Any]], float],
        is_higher_better: bool = True,
        halving_ratio: float = 0.5,
        min_candidates: int = 2000,
        params_stage_2: dict[str, Any],
        n_best_features: int = 100,
        save_dir: str | Path | None = None,
    ):
        assert 0 < max_order, "max_order must be positive"
        assert 0 < n_subsamples, "n_subsamples must be positive"
        assert 0 < halving_ratio <= 1, "halving_ratio must be in (0, 1]"
        assert 0 < min_candidates, "min_candidates must be positive"

        self.label = label
        self.max_order = max_order
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.n_subsamples = n_subsamples
        self.n_subsamples = n_subsamples
        self.params_stage_1 = params_stage_1
        self.init_score = init_score
        self.metric_fn = metric_fn
        self.is_higher_better = is_higher_better
        self.halving_ratio = halving_ratio
        self.min_candidates = min_candidates
        self.params_stage_2 = params_stage_2
        self.n_best_features = n_best_features
        self.save_dir = Path(save_dir) if save_dir is not None else None

        self.new_features: list[op.Operator] = []

        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        assert isinstance(validation_data, DataFrame), (
            "validation_data must be DataFrame"
        )

        logger.info("Enumerating new features")
        new_features = self.enumerate_new_features()
        logger.info(f"Enumerated {len(new_features)} new features")

        if self.save_dir is not None:
            with open(self.save_dir / "candidates.json", "w") as f:
                json.dump(
                    {
                        str(f): f.order
                        for f in sorted(new_features, key=lambda x: x.order)
                    },
                    f,
                    indent=4,
                )

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
            list(new_features_metric.keys()),
            params=self.params_stage_2,
            topk=self.n_best_features,
        )

        self.new_features = [
            k for k, _ in sorted(new_features_metric.items(), key=lambda x: -x[1])
        ]
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
                        op.Inv(col),
                        op.Pow(col),
                        op.Log(col),
                        op.Sqrt(col),
                        op.Square(col),
                        op.Sigmoid(col),
                        op.Floor(col),
                        op.Residual(col),
                        op.Abs(col),
                    ]
                )

            for col in cat_features[order]:
                num_features[-1].append(op.GroupByFreq(col, probability=True))

            for lhs_order in range(order + 1):
                rhs_order = order - lhs_order
                if lhs_order == rhs_order:
                    for lhs, rhs in itertools.combinations(num_features[lhs_order], 2):
                        num_features[-1].extend(
                            [
                                op.Add(lhs, rhs),
                                op.Sub(lhs, rhs),
                                op.Sub(rhs, lhs),
                                op.Mul(lhs, rhs),
                                op.Div(lhs, rhs),
                                op.Div(rhs, lhs),
                                op.Min(lhs, rhs),
                                op.Max(lhs, rhs),
                            ]
                        )

                    for lhs, rhs in itertools.combinations(cat_features[lhs_order], 2):
                        cat_features[-1].append(op.Combine(lhs, rhs))
                        num_features[-1].append(
                            op.GroupByFreq(op.Combine(lhs, rhs), probability=True)
                        )
                else:
                    for lhs, rhs in itertools.product(
                        num_features[lhs_order], num_features[rhs_order]
                    ):
                        num_features[-1].extend(
                            [
                                op.Add(lhs, rhs),
                                op.Sub(lhs, rhs),
                                op.Mul(lhs, rhs),
                                op.Div(lhs, rhs),
                                op.Min(lhs, rhs),
                                op.Max(lhs, rhs),
                            ]
                        )

                    for lhs, rhs in itertools.product(
                        cat_features[lhs_order], cat_features[rhs_order]
                    ):
                        cat_features[-1].append(op.Combine(lhs, rhs))
                        num_features[-1].append(
                            op.GroupByFreq(op.Combine(lhs, rhs), probability=True)
                        )

                for num, cat in itertools.product(
                    num_features[lhs_order], cat_features[rhs_order]
                ):
                    if num != cat:
                        num_features[-1].extend(
                            [
                                op.GroupByMean(cat, num),
                                op.GroupByStd(cat, num),
                                op.GroupByMin(cat, num),
                                op.GroupByQuantile(cat, num, quantile=0.25),
                                op.GroupByQuantile(cat, num, quantile=0.5),
                                op.GroupByQuantile(cat, num, quantile=0.75),
                                op.GroupByMax(cat, num),
                            ]
                        )

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
        logger.info("Screening new features with stage 1")

        valid_y = valid_data[label].to_numpy()
        scores: dict[op.Operator, float] = {}
        for iter, indexes in enumerate(
            incremental_sampling(train_data.height, self.n_subsamples)
        ):
            logger.info(f"Screening with {len(indexes)} samples")

            train_sub_df = train_data.select(pl.all().gather(indexes))
            train_sub_y = train_sub_df[label].to_numpy()
            train_sub_init_score = train_init_score[indexes]

            scores = {}
            for new_feature in tqdm(new_features, leave=False):
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

                # pred_valid_y: NDArray[Any] = gbm.predict(new_valid_X)  # type: ignore
                # curr_metric = self.metric_fn(valid_y, pred_valid_y)
                key = list(gbm.best_score["valid_0"].keys())[0]
                curr_metric = gbm.best_score["valid_0"][key]
                scores[new_feature] = curr_metric

            scores = deduplicate_scores(scores)
            next_features = [
                k
                for k, _ in sorted(
                    scores.items(),
                    key=lambda x: -x[1] if self.is_higher_better else x[1],
                )
            ]
            next_features = next_features[
                : max(
                    int(len(next_features) * self.halving_ratio),
                    min(self.min_candidates, len(next_features)),
                )
            ]
            new_features = next_features
            logger.info(f"Selected {len(new_features)} features")

            if self.save_dir is not None:
                with open(self.save_dir / f"stage_1_score_{iter}.json", "w") as f:
                    json.dump(
                        {
                            str(f): s
                            for f, s in sorted(
                                scores.items(),
                                key=lambda x: -x[1] if self.is_higher_better else x[1],
                            )
                        },
                        f,
                        indent=4,
                    )

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
        logger.info("Screening new features with stage 2")
        logger.info("Making new training features")

        new_train_data = pl.concat(
            [
                train_data,
                *[f.fit_transform(train_data) for f in tqdm(new_features, leave=False)],
            ],
            how="horizontal",
        )
        logger.info("Making new features finished")
        logger.info("Making new validation features")
        new_valid_data = pl.concat(
            [
                valid_data,
                *[f.transform(valid_data) for f in tqdm(new_features, leave=False)],
            ],
            how="horizontal",
        )
        logger.info("Making new features finished")

        train_X = new_train_data.drop(label).to_numpy()
        train_y = new_train_data[label].to_numpy()
        valid_X = new_valid_data.drop(label).to_numpy()
        valid_y = new_valid_data[label].to_numpy()

        lgb_train = lgb.Dataset(
            train_X, train_y, feature_name=new_train_data.drop(label).columns
        )
        lgb_valid = lgb_train.create_valid(valid_X, valid_y)

        logger.info("Training with all features")
        gbm = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_valid],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10)],
        )
        logger.info("Training finished")

        importance = gbm.feature_importance(importance_type="gain")
        feature_names = gbm.feature_name()

        name2op = {f.name: f for f in new_features}

        feature_importance = [
            (name, imp)
            for name, imp in zip(feature_names, importance)
            if name in name2op
        ]
        sorted_feature_importance = sorted(feature_importance, key=lambda x: -x[1])

        if self.save_dir is not None:
            with open(self.save_dir / "stage_2_score.json", "w") as f:
                json.dump(
                    {
                        str(name2op[name]): imp
                        for name, imp in sorted_feature_importance
                    },
                    f,
                    indent=4,
                )
        return {name2op[name]: imp for name, imp in sorted_feature_importance[:topk]}
