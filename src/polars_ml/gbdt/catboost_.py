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

import numpy as np
from numpy.typing import NDArray
from polars import DataFrame, Series
from polars._typing import IntoExpr

from polars_ml.pipeline.component import PipelineComponent

if TYPE_CHECKING:
    import catboost as cb

ObjectiveRegressionOptions = Literal[
    "RMSE",
    "MAE",
    "Quantile",
    "LogLinQuantile",
    "Poisson",
    "MAPE",
    "Huber",
    "Expectile",
    "Tweedie",
]
ObjectiveBinaryOptions = Literal[
    "Logloss",
    "CrossEntropy",
]
ObjectiveMulticlassOptions = Literal[
    "MultiClass",
    "MultiClassOneVsAll",
]
ObjectiveRankingOptions = Literal[
    "YetiRank",
    "YetiRankPairwise",
    "QueryRMSE",
    "QuerySoftMax",
    "PairLogit",
    "PairLogitPairwise",
    "QueryCrossEntropy",
]
ObjectiveOptions = (
    ObjectiveRegressionOptions
    | ObjectiveBinaryOptions
    | ObjectiveMulticlassOptions
    | ObjectiveRankingOptions
    | Literal["Custom"]
)

BoostingTypeOptions = Literal[
    "Ordered",
    "Plain",
]

GrowPolicyOptions = Literal[
    "SymmetricTree",
    "Depthwise",
    "Lossguide",
]

TaskTypeOptions = Literal[
    "CPU",
    "GPU",
]

BootstrapTypeOptions = Literal[
    "Bayesian",
    "Bernoulli",
    "MVS",
    "No",
]

NanModeOptions = Literal[
    "Min",
    "Max",
    "Forbidden",
]

LeafEstimationMethodOptions = Literal[
    "Newton",
    "Gradient",
]

ScoreFunctionOptions = Literal[
    "Cosine",
    "L2",
    "NewtonCosine",
    "NewtonL2",
]


class CatBoostParameters(TypedDict, total=False):
    loss_function: ObjectiveOptions
    custom_metric: list[str] | str
    eval_metric: str
    iterations: int
    learning_rate: float
    random_seed: int
    l2_leaf_reg: float
    bootstrap_type: BootstrapTypeOptions
    bagging_temperature: float
    subsample: float
    sampling_frequency: Literal["PerTree", "PerTreeLevel"]
    sampling_unit: Literal["Object", "Group"]
    random_strength: float
    use_best_model: bool
    best_model_min_trees: int
    depth: int
    grow_policy: GrowPolicyOptions
    min_data_in_leaf: int
    max_leaves: int
    ignored_features: list[int] | list[str]
    one_hot_max_size: int
    has_time: bool
    rsm: float
    nan_mode: NanModeOptions
    input_borders: str
    output_borders: str
    fold_permutation_block: int
    leaf_estimation_method: LeafEstimationMethodOptions
    leaf_estimation_iterations: int
    leaf_estimation_backtracking: Literal["No", "AnyImprovement", "Armijo"]
    fold_len_multiplier: float
    approx_on_full_history: bool
    class_weights: list[float]
    class_names: list[str]
    auto_class_weights: Literal["None", "Balanced", "SqrtBalanced"]
    scale_pos_weight: float
    boosting_type: BoostingTypeOptions
    boost_from_average: bool
    langevin: bool
    diffusion_temperature: float
    allow_const_label: bool
    score_function: ScoreFunctionOptions
    monotone_constraints: list[int]
    feature_weights: list[float]
    first_feature_use_penalties: list[float]
    penalties_coefficient: float
    model_shrink_rate: float
    model_shrink_mode: Literal["Constant", "Decreasing"]
    cat_features: list[int] | list[str]
    simple_ctr: list[str]
    combinations_ctr: list[str]
    per_feature_ctr: list[str]
    ctr_target_border_count: int
    counter_calc_method: Literal["SkipTest", "Full"]
    max_ctr_complexity: int
    ctr_leaf_count_limit: int
    store_all_simple_ctr: bool
    final_ctr_computation_mode: Literal["Default", "Skip"]
    logging_level: Literal["Silent", "Verbose", "Info", "Debug"]
    metric_period: int
    verbose: bool | int
    train_dir: str
    model_size_reg: float
    allow_writing_files: bool
    save_snapshot: bool
    snapshot_file: str
    snapshot_interval: int
    early_stopping_rounds: int
    od_type: Literal["IncToDec", "Iter"]
    od_pval: float
    od_wait: int
    thread_count: int
    used_ram_limit: str
    gpu_ram_part: float
    pinned_memory_size: int
    task_type: TaskTypeOptions
    devices: str
    border_count: int
    feature_border_type: Literal[
        "Median",
        "Uniform",
        "UniformAndQuantiles",
        "MaxLogSum",
        "MinEntropy",
        "GreedyLogSum",
    ]
    text_features: list[int] | list[str]
    tokenizers: list[dict[str, Any]]
    dictionaries: list[dict[str, Any]]
    feature_calcers: list[str]
    text_processing: dict[str, Any]
    n_estimators: int
    num_boost_round: int
    num_trees: int
    max_depth: int
    eta: float
    reg_lambda: float
    objective: ObjectiveOptions
    random_state: int
    colsample_bylevel: float
    min_child_samples: int
    num_leaves: int


CatBoostFEval = (
    Callable[[NDArray[Any], "cb.Pool"], tuple[str, float, bool]]
    | Callable[[NDArray[Any], "cb.Pool"], list[tuple[str, float, bool]]]
)


class CatBoostTrainArguments(TypedDict, total=False):
    eval_set: Union["cb.Pool", list["cb.Pool"]]
    verbose: bool | int
    logging_level: Literal["Silent", "Verbose", "Info", "Debug"]
    plot: bool
    plot_file: str
    metric_period: int
    early_stopping_rounds: int
    save_snapshot: bool
    snapshot_file: str
    snapshot_interval: int
    init_model: Union[str, Path, "cb.CatBoost", None]


class CatBoostPredictArguments(TypedDict, total=False):
    verbose: bool | int
    thread_count: int
    task_type: TaskTypeOptions
    ntree_start: int
    ntree_end: int


class CatBoostPoolArguments(TypedDict, total=False):
    cat_features: list[int] | list[str]
    text_features: list[int] | list[str]
    embedding_features: list[int] | list[str]
    weight: list[float] | NDArray[Any]
    feature_names: list[str]


class CatBoost(PipelineComponent):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        params: CatBoostParameters | None = None,
        *,
        prediction_name: str = "catboost",
        include_input: bool = True,
        train_kwargs: CatBoostTrainArguments
        | Callable[[DataFrame], CatBoostTrainArguments]
        | None = None,
        predict_kwargs: CatBoostPredictArguments
        | Callable[[DataFrame, "cb.CatBoost"], CatBoostPredictArguments]
        | None = None,
        pool_kwargs: CatBoostPoolArguments
        | Callable[[DataFrame], CatBoostPoolArguments]
        | None = None,
        out_dir: str | Path | None = None,
    ):
        self.features = features
        self.label = label
        self.params = params or {}
        self.prediction_name = prediction_name
        self.include_input = include_input
        self.train_kwargs = train_kwargs or {}
        self.predict_kwargs = predict_kwargs or {}
        self.pool_kwargs = pool_kwargs or {}
        self.out_dir = Path(out_dir) if out_dir is not None else None

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        import catboost as cb

        train_features = data.select(self.features)
        train_label = data.select(self.label)
        self.feature_names = train_features.columns

        pool_kwargs = (
            self.pool_kwargs(data) if callable(self.pool_kwargs) else self.pool_kwargs
        )

        pool_kwargs["feature_names"] = train_features.columns

        train_pool = cb.Pool(
            data=train_features.to_numpy(),
            label=train_label.to_numpy().squeeze(),
            **pool_kwargs,
        )

        eval_set = None
        if validation_data is not None:
            if isinstance(validation_data, DataFrame):
                valid_features = validation_data.select(self.features)
                valid_label = validation_data.select(self.label)
                valid_pool = cb.Pool(
                    data=valid_features.to_numpy(),
                    label=valid_label.to_numpy().squeeze(),
                    **pool_kwargs,
                )
                eval_set = valid_pool
            else:
                eval_pools = []
                for _, raw_valid_data in validation_data.items():
                    valid_features = raw_valid_data.select(self.features)
                    valid_label = raw_valid_data.select(self.label)
                    valid_pool = cb.Pool(
                        data=valid_features.to_numpy(),
                        label=valid_label.to_numpy().squeeze(),
                        **pool_kwargs,
                    )
                    eval_pools.append(valid_pool)
                eval_set = eval_pools

        train_kwargs = (
            self.train_kwargs(data)
            if callable(self.train_kwargs)
            else self.train_kwargs
        )

        if "eval_set" not in train_kwargs and eval_set is not None:
            train_kwargs["eval_set"] = eval_set

        self.model = cb.train(params=self.params, dtrain=train_pool, **train_kwargs)

        if self.out_dir is not None:
            self.save()

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        input_data = data.select(self.features)
        predict_kwargs = (
            self.predict_kwargs(data, self.model)
            if callable(self.predict_kwargs)
            else self.predict_kwargs
        )

        pred: NDArray[Any] = self.model.predict(input_data.to_numpy(), **predict_kwargs)

        if pred.ndim == 1:
            columns = [Series(self.prediction_name, pred)]
        else:
            columns = [
                Series(f"{self.prediction_name}_{i}", pred[:, i])
                for i in range(pred.shape[1])
            ]

        if self.include_input:
            return data.with_columns(columns)
        else:
            return DataFrame(columns)

    def save(self, out_dir: str | Path | None = None):
        import matplotlib.pyplot as plt

        out_dir = Path(out_dir) if out_dir is not None else self.out_dir
        if out_dir is None:
            raise ValueError("No output directory provided")

        out_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(out_dir / "model.cbm"))

        feature_importances = self.model.get_feature_importance()

        if feature_importances is not None:
            n_features = len(self.feature_names)
            indices = np.arange(n_features)

            plt.figure(figsize=(10, 6))
            plt.barh(indices, feature_importances)  # type: ignore
            plt.yticks(indices, self.feature_names)
            plt.xlabel("Feature Importance")
            plt.tight_layout()
            plt.savefig(str(out_dir / "feature_importance.png"))
            plt.close()

        for i in range(min(5, self.model.tree_count_ or 0)):
            self.model.plot_tree(i)
            plt.savefig(out_dir / f"tree_{i}.png")
            plt.close()
