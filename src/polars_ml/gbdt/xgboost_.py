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

from numpy.typing import NDArray
from polars import DataFrame, Series
from polars._typing import IntoExpr

from polars_ml.pipeline.component import PipelineComponent

if TYPE_CHECKING:
    import xgboost as xgb

ObjectiveRegressionOptions = Literal[
    "reg:squarederror",
    "reg:squaredlogerror",
    "reg:logistic",
    "reg:pseudohubererror",
    "reg:absoluteerror",
    "reg:quantileerror",
    "count:poisson",
    "reg:gamma",
    "reg:tweedie",
]
ObjectiveBinaryOptions = Literal[
    "binary:logistic",
    "binary:logitraw",
    "binary:hinge",
]
ObjectiveMulticlassOptions = Literal[
    "multi:softmax",
    "multi:softprob",
]
ObjectiveRankingOptions = Literal[
    "rank:ndcg",
    "rank:map",
    "rank:pairwise",
]
ObjectiveSurvivalOptions = Literal[
    "survival:cox",
    "survival:aft",
]
ObjectiveOptions = (
    ObjectiveRegressionOptions
    | ObjectiveBinaryOptions
    | ObjectiveMulticlassOptions
    | ObjectiveRankingOptions
    | ObjectiveSurvivalOptions
    | Literal["custom"]
)

TreeMethodOptions = Literal[
    "auto",
    "exact",
    "approx",
    "hist",
]

BoosterOptions = Literal[
    "gbtree",
    "gblinear",
    "dart",
]

DeviceOptions = Literal[
    "cpu",
    "cuda",
    "gpu",
]

GrowPolicyOptions = Literal[
    "depthwise",
    "lossguide",
]


class XGBoostParameters(TypedDict, total=False):
    objective: ObjectiveOptions
    booster: BoosterOptions
    device: DeviceOptions
    tree_method: TreeMethodOptions
    n_estimators: int
    learning_rate: float
    gamma: float
    min_child_weight: float
    max_depth: int
    max_delta_step: float
    subsample: float
    sampling_method: Literal["uniform", "gradient_based"]
    colsample_bytree: float
    colsample_bylevel: float
    colsample_bynode: float
    reg_lambda: float
    reg_alpha: float
    scale_pos_weight: float
    grow_policy: GrowPolicyOptions
    max_leaves: int
    max_bin: int
    num_parallel_tree: int
    random_state: int | None

    sample_type: Literal["uniform", "weighted"]
    normalize_type: Literal["tree", "forest"]
    rate_drop: float
    one_drop: bool
    skip_drop: float

    monotone_constraints: list[int] | None
    interaction_constraints: list[list[int]] | None
    verbosity: int
    validate_parameters: bool
    nthread: int
    disable_default_eval_metric: bool
    base_score: float
    eval_metric: str | list[str]
    early_stopping_rounds: int
    num_class: int
    seed: int

    tweedie_variance_power: float

    huber_slope: float

    quantile_alpha: float | list[float]

    aft_loss_distribution: Literal["normal", "logistic", "extreme"]

    lambdarank_pair_method: Literal["mean", "topk"]
    lambdarank_num_pair_per_sample: int
    lambdarank_normalization: bool
    lambdarank_score_normalization: bool
    lambdarank_unbiased: bool
    lambdarank_bias_norm: float
    ndcg_exp_gain: bool


XGBoostFEval = (
    Callable[[NDArray[Any], "xgb.DMatrix"], tuple[str, float, bool]]
    | Callable[[NDArray[Any], "xgb.DMatrix"], list[tuple[str, float, bool]]]
)


class XGBoostTrainArguments(TypedDict, total=False):
    num_boost_round: int
    feval: XGBoostFEval | list[XGBoostFEval] | None
    evals: list[tuple["xgb.DMatrix", str]] | None
    obj: Callable[..., Any] | None
    verbose_eval: bool | int
    xgb_model: Union[str, Path, "xgb.Booster", None]
    callbacks: list[Callable[..., Any]] | None
    early_stopping_rounds: int


class XGBoostPredictArguments(TypedDict, total=False):
    iteration_range: tuple[int, int] | None
    pred_leaf: bool
    pred_contribs: bool
    approx_contribs: bool
    pred_interactions: bool
    output_margin: bool
    validate_features: bool
    training: bool
    strict_shape: bool


class XGBoostDMatrixArguments(TypedDict, total=False):
    weight: NDArray[Any] | None
    base_margin: NDArray[Any] | None
    missing: float | None
    silent: bool
    feature_types: list[str] | None
    nthread: int | None
    group: NDArray[Any] | None
    qid: NDArray[Any] | None
    label_lower_bound: NDArray[Any] | None
    label_upper_bound: NDArray[Any] | None
    feature_weights: NDArray[Any] | None
    enable_categorical: bool


class XGBoost(PipelineComponent):
    def __init__(
        self,
        features: IntoExpr | Iterable[IntoExpr],
        label: IntoExpr,
        params: XGBoostParameters | None = None,
        *,
        prediction_name: str = "xgboost",
        include_input: bool = True,
        train_kwargs: XGBoostTrainArguments
        | Callable[[DataFrame], XGBoostTrainArguments]
        | None = None,
        predict_kwargs: XGBoostPredictArguments
        | Callable[[DataFrame, "xgb.Booster"], XGBoostPredictArguments]
        | None = None,
        dmatrix_kwargs: XGBoostDMatrixArguments
        | Callable[[DataFrame], XGBoostDMatrixArguments]
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
        self.dmatrix_kwargs = dmatrix_kwargs or {}
        self.out_dir = Path(out_dir) if out_dir is not None else None

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        import xgboost as xgb

        train_features = data.select(self.features)
        train_label = data.select(self.label)
        self.feature_names = train_features.columns

        dmatrix_kwargs = (
            self.dmatrix_kwargs(data)
            if callable(self.dmatrix_kwargs)
            else self.dmatrix_kwargs
        )
        train_dmatrix = xgb.DMatrix(
            train_features.to_numpy(),
            label=train_label.to_numpy().squeeze(),
            feature_names=train_features.columns,
            **dmatrix_kwargs,
        )

        evals = []
        if validation_data is not None:
            if isinstance(validation_data, DataFrame):
                valid_features = validation_data.select(self.features)
                valid_label = validation_data.select(self.label)
                valid_dmatrix = xgb.DMatrix(
                    valid_features.to_numpy(),
                    label=valid_label.to_numpy().squeeze(),
                    feature_names=valid_features.columns,
                    **dmatrix_kwargs,
                )
                evals.append((valid_dmatrix, "valid"))
            else:
                for name, raw_valid_data in validation_data.items():
                    valid_features = raw_valid_data.select(self.features)
                    valid_label = raw_valid_data.select(self.label)
                    valid_dmatrix = xgb.DMatrix(
                        valid_features.to_numpy(),
                        label=valid_label.to_numpy().squeeze(),
                        feature_names=valid_features.columns,
                        **dmatrix_kwargs,
                    )
                    evals.append((valid_dmatrix, name))

        evals.append((train_dmatrix, "train"))

        train_kwargs = (
            self.train_kwargs(data)
            if callable(self.train_kwargs)
            else self.train_kwargs
        )

        self.model = xgb.train(
            self.params,
            train_dmatrix,
            **train_kwargs,
        )

        if self.out_dir is not None:
            self.save()

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        import xgboost as xgb

        input = data.select(self.features)
        predict_kwargs = (
            self.predict_kwargs(data, self.model)
            if callable(self.predict_kwargs)
            else self.predict_kwargs
        )

        dmatrix_kwargs = (
            self.dmatrix_kwargs(data)
            if callable(self.dmatrix_kwargs)
            else self.dmatrix_kwargs
        )
        dmatrix = xgb.DMatrix(
            input.to_numpy(),
            feature_names=input.columns,
            **{k: v for k, v in dmatrix_kwargs.items() if k != "label"},
        )

        pred: NDArray[Any] = self.model.predict(dmatrix, **predict_kwargs)  # type: ignore

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

        out_dir = Path(out_dir) if out_dir else self.out_dir
        if out_dir is None:
            raise ValueError("No output directory provided")

        out_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_model(str(out_dir / "model.json"))

        for importance_type in [
            "weight",
            "gain",
            "cover",
            "total_gain",
            "total_cover",
        ]:
            xgb.plot_importance(self.model, importance_type=importance_type)
            plt.tight_layout()
            plt.savefig(out_dir / f"importance_{importance_type}.png")
            plt.close()

        for i in range(min(5, self.model.num_boosted_rounds())):
            xgb.plot_tree(self.model, num_trees=i)
            plt.savefig(out_dir / f"tree_{i}.png")
            plt.close()
