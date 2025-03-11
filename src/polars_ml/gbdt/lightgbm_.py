from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Literal,
    Mapping,
    Self,
    TypedDict,
    Union,
)

import polars as pl
from numpy.typing import NDArray
from polars import DataFrame
from polars._typing import IntoExpr

from polars_ml.pipeline.component import PipelineComponent

if TYPE_CHECKING:
    import lightgbm as lgb

ObjectiveRegressionOptions = Literal[
    "regression",
    "regression_l2",
    "l2",
    "mean_squared_error",
    "mse",
    "l2_root",
    "root_mean_squared_error",
    "rmse",
    "regression_l1",
    "l1",
    "mean_absolute_error",
    "mae",
    "huber",
    "fair",
    "poisson",
    "quantile",
    "mape",
    "mean_absolute_percentage_error",
    "gamma",
    "tweedie",
]
ObjectiveBinaryOptions = Literal[
    "binary",
    "cross_entropy",
    "xentropy",
    "cross_entropy_lambda",
    "xentlambda",
]
ObjectiveMulticlassOptions = Literal[
    "multiclass", "multiclassova", "multiclass_ova", "ova", "ovr"
]
ObjectiveRankingOptions = Literal[
    "lambdarank",
    "rank_xendcg",
    "xendcg",
    "xe_ndcg",
    "xe_ndcg_mart",
    "xendcg_mart",
]
ObjectiveOptions = (
    ObjectiveRegressionOptions
    | ObjectiveBinaryOptions
    | ObjectiveMulticlassOptions
    | ObjectiveRankingOptions
    | Literal["custom"]
)

TreeLearnerOptions = Literal[
    "serial",
    "feature",
    "feature_parallel",
    "data",
    "data_parallel",
    "voting",
    "voting_parallel",
]


class LightGBMParameters(TypedDict, total=False):
    objective: ObjectiveOptions
    boosting: Literal["gbdt", "rf", "dart"]
    data_sample_strategy: Literal["bagging", "goss"]
    num_iterations: int
    learning_rate: float
    num_leaves: int
    tree_learner: TreeLearnerOptions
    num_threads: int
    device_type: Literal["cpu", "gpu", "cuda"]
    seed: int | None
    deterministic: bool

    force_col_wise: bool
    force_row_wise: bool
    histogram_pool_size: float
    max_depth: int
    min_data_in_leaf: int
    min_sum_hessian_in_leaf: float
    bagging_fraction: float
    pos_bagging_fraction: float
    neg_bagging_fraction: float
    bagging_freq: int
    bagging_seed: int
    feature_fraction: float
    feature_fraction_bynode: float
    feature_fraction_seed: int
    extra_trees: bool
    extra_seed: int
    early_stopping_round: int
    early_stopping_min_delta: float
    first_metric_only: bool
    max_delta_step: float
    lambda_l1: float
    lambda_l2: float
    linear_lambda: float
    min_gain_to_split: float
    drop_rate: float
    max_drop: int
    skip_drop: float
    xgboost_dart_mode: bool
    uniform_drop: bool
    drop_seed: int
    top_rate: float
    other_rate: float
    min_data_per_group: int
    max_cat_threshold: int
    cat_l2: float
    cat_smooth: float
    max_cat_to_onehot: int
    top_k: int
    monotone_constraints: list[int] | None
    monotone_constraints_method: Literal["basic", "intermediate", "advanced"]
    monotone_penalty: float
    feature_contri: list[float] | None
    forcedsplits_filename: str
    refit_decay_rate: float
    cegb_tradeoff: float
    cegb_penalty_split: float
    path_smooth: float
    verbosity: int
    is_provide_training_metric: bool
    eval_at: list[int]
    metric_freq: int
    linear_tree: bool
    max_bin: int
    min_data_in_bin: int
    bin_construct_sample_cnt: int
    data_random_seed: int
    is_enable_sparse: bool
    enable_bundle: bool
    use_missing: bool
    zero_as_missing: bool
    feature_pre_filter: bool
    num_class: int
    is_unbalance: bool
    scale_pos_weight: float
    sigmoid: float
    boost_from_average: bool
    reg_sqrt: bool
    alpha: float
    fair_c: float
    poisson_max_delta_step: float
    tweedie_variance_power: float
    lambdarank_truncation_level: int
    lambdarank_norm: bool
    label_gain: list[float] | None
    gpu_platform_id: int
    gpu_device_id: int
    cgpu_use_dp: bool
    num_gpu: int


LightGBMFEval = (
    Callable[[NDArray[Any], "lgb.Dataset"], tuple[str, float, bool]]
    | Callable[[NDArray[Any], "lgb.Dataset"], list[tuple[str, float, bool]]]
)


class LightGBMTrainArguments(TypedDict, total=False):
    num_boost_round: int
    feval: LightGBMFEval | list[LightGBMFEval] | None
    init_model: Union[str, Path, "lgb.Booster", None]
    keep_training_booster: bool
    callbacks: list[Callable[..., Any]] | None


class LightGBMPredictArguments(TypedDict, total=False):
    start_iteration: int
    num_iteration: int | None
    raw_score: bool
    pred_leaf: bool
    pred_contrib: bool
    data_has_header: bool
    validate_features: bool
    kwargs: Mapping[str, Any]


class LightGBMTrainDatasetArguments(TypedDict, total=False):
    weight: NDArray[Any] | None
    group: NDArray[Any] | None
    init_score: NDArray[Any] | None
    categorical_feature: list[str] | list[int] | Literal["auto"]
    params: dict[str, Any] | None
    free_raw_data: bool
    position: NDArray[Any] | None


class LightGBMValidateDatasetArguments(TypedDict, total=False):
    weight: NDArray[Any] | None
    group: NDArray[Any] | None
    init_score: NDArray[Any] | None
    params: dict[str, Any] | None
    position: NDArray[Any] | None


class LightGBM(PipelineComponent):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        params: LightGBMParameters | None = None,
        *,
        prediction_name: str = "prediction",
        append_prediction: bool = True,
        train_kwargs: LightGBMTrainArguments
        | Callable[[DataFrame], LightGBMTrainArguments]
        | None = None,
        predict_kwargs: LightGBMPredictArguments
        | Callable[[DataFrame, "lgb.Booster"], LightGBMPredictArguments]
        | None = None,
        train_dataset_kwargs: LightGBMTrainDatasetArguments
        | Callable[[DataFrame], LightGBMTrainDatasetArguments]
        | None = None,
        validation_dataset_kwargs: LightGBMValidateDatasetArguments
        | Callable[[DataFrame], LightGBMValidateDatasetArguments]
        | None = None,
        save_dir: str | Path | None = None,
    ):
        self.features = features
        self.label = label
        self.params = params or {}
        self.prediction_name = prediction_name
        self.append_prediction = append_prediction
        self.train_kwargs = train_kwargs or {}
        self.predict_kwargs = predict_kwargs or {}
        self.train_dataset_kwargs = train_dataset_kwargs or {}
        self.validation_dataset_kwargs = validation_dataset_kwargs or {}
        self.save_dir = Path(save_dir) if save_dir is not None else None

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        import lightgbm as lgb

        train_features = data.select(self.features)
        train_label = data.select(self.label)
        train_dataset_kwargs = (
            self.train_dataset_kwargs(data)
            if callable(self.train_dataset_kwargs)
            else self.train_dataset_kwargs
        )
        train_dataset = lgb.Dataset(
            train_features.to_numpy(),
            label=train_label.to_numpy().squeeze(),
            feature_name=train_features.columns,
            **train_dataset_kwargs,
        )

        valid_sets = []
        valid_names = []
        if validation_data is not None:
            if isinstance(validation_data, DataFrame):
                valid_features = validation_data.select(self.features)
                valid_label = validation_data.select(self.label)

                valid_dataset_kwargs = (
                    self.validation_dataset_kwargs(validation_data)
                    if callable(self.validation_dataset_kwargs)
                    else self.validation_dataset_kwargs
                )
                valid_dataset = train_dataset.create_valid(
                    valid_features.to_numpy(),
                    label=valid_label.to_numpy().squeeze(),
                    **valid_dataset_kwargs,
                )
                valid_sets.append(valid_dataset)
                valid_names.append("valid")
            else:
                for name, raw_valid_data in validation_data.items():
                    valid_features = raw_valid_data.select(self.features)
                    valid_label = raw_valid_data.select(self.label)
                    valid_dataset_kwargs = (
                        self.validation_dataset_kwargs(raw_valid_data)
                        if callable(self.validation_dataset_kwargs)
                        else self.validation_dataset_kwargs
                    )
                    valid_dataset = train_dataset.create_valid(
                        valid_features.to_numpy(),
                        label=valid_label.to_numpy().squeeze(),
                        **valid_dataset_kwargs,
                    )
                    valid_sets.append(valid_dataset)
                    valid_names.append(name)

        valid_sets.append(train_dataset)
        valid_names.append("train")

        train_kwargs = (
            self.train_kwargs(data)
            if callable(self.train_kwargs)
            else self.train_kwargs
        )

        print(self.params, train_kwargs)
        self.model = lgb.train(
            dict(**self.params),
            train_dataset,
            valid_sets=valid_sets,
            valid_names=valid_names,
            **train_kwargs,
        )

        if self.save_dir is not None:
            import matplotlib.pyplot as plt

            self.save_dir.mkdir(parents=True, exist_ok=True)

            self.model.save_model(self.save_dir / "model.txt")

            for importance_type in ["gain", "split"]:
                lgb.plot_importance(self.model, importance_type=importance_type)
                plt.tight_layout()
                plt.savefig(self.save_dir / f"importance_{importance_type}.png")
                plt.close()

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input = data.select(self.features)
        predict_kwargs = (
            self.predict_kwargs(data, self.model)
            if callable(self.predict_kwargs)
            else self.predict_kwargs
        )
        pred: NDArray[Any] = self.model.predict(input.to_numpy(), **predict_kwargs)  # type: ignore
        if pred.ndim == 1:
            schema = [self.prediction_name]
        else:
            schema = [f"{self.prediction_name}_{i}" for i in range(pred.shape[1])]

        pred_df = pl.from_numpy(pred, schema=schema)
        if self.append_prediction:
            return pl.concat([data, pred_df], how="horizontal")
        else:
            return pred_df
