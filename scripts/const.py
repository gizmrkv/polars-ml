from __future__ import annotations

from polars_ml.feature_engineering import (
    ArithmeticSynthesis,
    FeatureEngineeringNameSpace,
)
from polars_ml.gbdt import CatBoost, LightGBM, LightGBMTuner, LightGBMTunerCV, XGBoost
from polars_ml.metrics import (
    BinaryClassificationMetrics,
    MulticlassClassificationMetrics,
    RegressionMetrics,
)
from polars_ml.optimize import LinearEnsemble, OptunaOptimizer, WeightedAverage
from polars_ml.pipeline.basic import Apply, Concat, Const, Echo, Replay, Side, ToDummies
from polars_ml.pipeline.group_by import (
    DynamicGroupByNameSpace,
    GroupByNameSpace,
    RollingGroupByNameSpace,
)
from polars_ml.preprocessing import (
    AggJoin,
    BoxCoxTransform,
    Combine,
    Discretize,
    HorizontalNameSpace,
    LabelEncode,
    LabelEncodeInverseContext,
    MinMaxScale,
    PowerTransformInverseContext,
    RobustScale,
    ScaleInverseContext,
    StandardScale,
    YeoJohnsonTransform,
)

START_INSERTION_MARKER = "{prefix}# --- START INSERTION MARKER{suffix}"
END_INSERTION_MARKER = "{prefix}# --- END INSERTION MARKER{suffix}"

BUILT_IN_FUNCTION_BLOCK_LIST = {
    "map_columns",
    "deserialize",
    "to_dummies",
    "get_column",
    "write_delta",
    "write_excel",
    "write_iceberg",
    "read_delta",
    "read_excel",
    "read_database",
}
GROUP_BY_NAMESPACES: list[tuple[str, type]] = [
    ("group_by", GroupByNameSpace),
    ("group_by_dynamic", DynamicGroupByNameSpace),
    ("rolling", RollingGroupByNameSpace),
]
HORIZONTAL_NAMESPACES: list[tuple[str, type]] = [
    ("horizontal", HorizontalNameSpace),
]
FEATURE_ENGINEERING_NAMESPACES: list[tuple[str, type]] = [
    ("feature_engineering", FeatureEngineeringNameSpace),
]
BASIC_TRANSFORMERS: list[type] = [
    Apply,
    Const,
    Echo,
    Replay,
    Side,
    Discretize,
    Combine,
    Concat,
    ToDummies,
    AggJoin,
]
BASIC_TRANSFORMERS_WITH_INVERSE: list[tuple[type, type]] = [
    (MinMaxScale, ScaleInverseContext),
    (StandardScale, ScaleInverseContext),
    (RobustScale, ScaleInverseContext),
    (BoxCoxTransform, PowerTransformInverseContext),
    (YeoJohnsonTransform, PowerTransformInverseContext),
    (LabelEncode, LabelEncodeInverseContext),
]
GBDT_TRANSFORMERS: list[tuple[str, type]] = [
    ("lightgbm", LightGBM),
    ("xgboost", XGBoost),
    ("lightgbm_tuner", LightGBMTuner),
    ("lightgbm_tuner_cv", LightGBMTunerCV),
    ("catboost", CatBoost),
]
METRICS_TRANSFORMERS: list[tuple[str, type]] = [
    ("binary_classification", BinaryClassificationMetrics),
    ("multiclass_classification", MulticlassClassificationMetrics),
    ("regression", RegressionMetrics),
]
OPTIMIZE_TRANSFORMERS: list[tuple[str, type]] = [
    ("optuna", OptunaOptimizer),
    ("weighted_average", WeightedAverage),
    ("linear_ensemble", LinearEnsemble),
]
