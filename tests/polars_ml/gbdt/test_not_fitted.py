import pytest
from polars import DataFrame

from polars_ml.exceptions import NotFittedError
from polars_ml.gbdt.catboost_ import CatBoost
from polars_ml.gbdt.lightgbm_ import LightGBM, LightGBMTuner, LightGBMTunerCV
from polars_ml.gbdt.xgboost_ import XGBoost


def test_lightgbm_not_fitted():
    model = LightGBM(params={}, label="target")
    df = DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]})
    with pytest.raises(NotFittedError):
        model.transform(df)


def test_lightgbm_tuner_not_fitted():
    model = LightGBMTuner(params={}, label="target")
    df = DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]})
    with pytest.raises(NotFittedError):
        model.transform(df)


def test_lightgbm_tunercv_not_fitted():
    model = LightGBMTunerCV(params={}, label="target")
    df = DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]})
    with pytest.raises(NotFittedError):
        model.transform(df)


def test_catboost_not_fitted():
    model = CatBoost(params={}, label="target")
    df = DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]})
    with pytest.raises(NotFittedError):
        model.transform(df)


def test_xgboost_not_fitted():
    model = XGBoost(params={}, label="target")
    df = DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]})
    with pytest.raises(NotFittedError):
        model.transform(df)
