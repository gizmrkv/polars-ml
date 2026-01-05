from polars import DataFrame
from sklearn.linear_model import LinearRegression as SkLinearRegression
from sklearn.linear_model import LogisticRegression as SkLogisticRegression

from polars_ml import Pipeline
from polars_ml.linear import LinearRegression, LogisticRegression


def test_linear_regression():
    df = DataFrame(
        {"x1": [1, 2, 3, 4, 5], "x2": [10, 20, 30, 40, 50], "y": [2, 4, 6, 8, 10]}
    )

    model = SkLinearRegression()
    lr = LinearRegression(model, label="y", features=["x1", "x2"])

    lr.fit(df)
    result = lr.transform(df)

    assert result.columns == ["prediction"]
    assert len(result) == 5
    # Since it's a perfect linear relation, predictions should be very close to y
    assert (result["prediction"] - df["y"]).abs().max() < 1e-10


def test_logistic_regression():
    df = DataFrame({"x1": [1, 2, 3, 4, 5], "y": [0, 0, 1, 1, 1]})

    model = SkLogisticRegression()
    lr = LogisticRegression(model, label="y")

    lr.fit(df)
    result = lr.transform(df)

    # Logistic regression returns probabilities (2 columns for binary)
    assert result.columns == ["prediction_0", "prediction_1"]
    assert len(result) == 5
    assert (
        (result["prediction_0"] + result["prediction_1"]).is_between(0.999, 1.001).all()
    )


def test_pipeline_integration():
    df = DataFrame({"x1": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})

    pipeline = Pipeline().linear.regression(
        SkLinearRegression(), label="y", prediction_name="val"
    )

    pipeline.fit(df)
    result = pipeline.transform(df)

    assert result.columns == ["val"]
    assert len(result) == 5


def test_feature_selection_exclude_label():
    df = DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6], "target": [7, 8, 9]})

    lr = LinearRegression(SkLinearRegression(), label="target")
    lr.fit(df)

    assert set(lr.feature_names) == {"f1", "f2"}
