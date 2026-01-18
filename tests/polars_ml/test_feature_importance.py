import pytest
from polars import DataFrame
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import LogisticRegression as SkLogisticRegression

from polars_ml.base import HasFeatureImportance
from polars_ml.gbdt import CatBoost, LightGBM, XGBoost
from polars_ml.linear import LinearRegression, LogisticRegression


def test_lightgbm_feature_importance() -> None:
    df = DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6], "target": [0, 1, 0]})
    model = LightGBM({"verbosity": -1}, label="target")
    model.fit(df)

    assert isinstance(model, HasFeatureImportance)
    importance = model.get_feature_importance()
    assert isinstance(importance, DataFrame)
    assert set(importance.columns) == {"feature", "gain", "split"}
    assert set(importance["feature"]) == {"f1", "f2"}


def test_xgboost_feature_importance() -> None:
    df = DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6], "target": [0, 1, 0]})
    model = XGBoost({"verbosity": 0}, label="target")
    model.fit(df)

    assert isinstance(model, HasFeatureImportance)
    importance = model.get_feature_importance()
    assert isinstance(importance, DataFrame)
    assert set(importance.columns) == {
        "feature",
        "gain",
        "weight",
        "cover",
        "total_gain",
        "total_cover",
    }
    assert set(importance["feature"]) == {"f1", "f2"}


def test_catboost_feature_importance() -> None:
    df = DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6], "target": [0, 1, 0]})
    model = CatBoost({"verbose": False}, label="target")
    model.fit(df)

    assert isinstance(model, HasFeatureImportance)
    importance = model.get_feature_importance()
    assert isinstance(importance, DataFrame)
    assert set(importance.columns) == {"feature", "importance"}
    assert set(importance["feature"]) == {"f1", "f2"}


def test_linear_regression_importance() -> None:
    df = DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6], "target": [7, 8, 9]})
    model = SkLinearRegression()
    lr = LinearRegression(model, label="target")
    lr.fit(df)

    assert isinstance(lr, HasFeatureImportance)
    importance = lr.get_feature_importance()
    assert isinstance(importance, DataFrame)
    assert set(importance.columns) == {"feature", "coefficient"}
    assert set(importance["feature"]) == {"f1", "f2"}


def test_logistic_regression_importance_binary() -> None:
    df = DataFrame({"f1": [1, 2, 3, 4], "target": [0, 0, 1, 1]})
    model = SkLogisticRegression()
    lr = LogisticRegression(model, label="target")
    lr.fit(df)

    assert isinstance(lr, HasFeatureImportance)
    importance = lr.get_feature_importance()
    assert isinstance(importance, DataFrame)
    assert set(importance.columns) == {"feature", "coefficient_class_0"}
    assert set(importance["feature"]) == {"f1"}


def test_logistic_regression_importance_multi() -> None:
    df = DataFrame({"f1": [1, 2, 3, 4, 5, 6], "target": [0, 0, 1, 1, 2, 2]})
    model = SkLogisticRegression()
    lr = LogisticRegression(model, label="target")
    lr.fit(df)

    assert isinstance(lr, HasFeatureImportance)
    importance = lr.get_feature_importance()
    assert isinstance(importance, DataFrame)
    assert "coefficient_class_0" in importance.columns
    assert "coefficient_class_1" in importance.columns
    assert "coefficient_class_2" in importance.columns
    assert set(importance["feature"]) == {"f1"}


def test_pipeline_feature_importance() -> None:
    from polars_ml import Pipeline

    df = DataFrame({"f1": [1, 2, 3], "target": [7, 8, 9]})
    pipeline = Pipeline().linear.regression(SkLinearRegression(), label="target")
    pipeline.fit(df)

    assert isinstance(pipeline, HasFeatureImportance)
    importance = pipeline.get_feature_importance()
    assert isinstance(importance, DataFrame)
    assert set(importance.columns) == {"feature", "coefficient"}


def test_pipeline_feature_importance_error() -> None:
    from polars_ml import Pipeline
    from polars_ml.pipeline.basic import Echo

    df = DataFrame({"f1": [1, 2, 3], "target": [7, 8, 9]})
    # Echo does not support feature importance
    pipeline = Pipeline(Echo())
    pipeline.fit(df)

    with pytest.raises(TypeError, match="does not support feature importance"):
        pipeline.get_feature_importance()
