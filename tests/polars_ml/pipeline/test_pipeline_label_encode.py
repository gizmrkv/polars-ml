import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal

from polars_ml.exceptions import NotFittedError
from polars_ml.pipeline.pipeline import Pipeline


def test_label_encode_basic():
    df = DataFrame({"cat": ["a", "b", "a", "c"]})

    pipeline = Pipeline().label_encode("cat")

    result = pipeline.fit_transform(df)

    expected = df.with_columns(cat=pl.Series([0, 1, 0, 2], dtype=pl.UInt32))
    assert_frame_equal(result, expected)


def test_label_encode_multiple_cols():
    df = DataFrame({"c1": ["x", "y"], "c2": ["m", "n"]})

    pipeline = Pipeline().label_encode("c1", "c2")

    result = pipeline.fit_transform(df)

    expected = df.with_columns(
        c1=pl.Series([0, 1], dtype=pl.UInt32), c2=pl.Series([0, 1], dtype=pl.UInt32)
    )
    assert_frame_equal(result, expected)


def test_label_encode_custom_orders():
    df = DataFrame({"cat": ["a", "b"]})

    pipeline = Pipeline().label_encode("cat", orders={"cat": ["b", "a"]})

    result = pipeline.fit_transform(df)

    expected = df.with_columns(cat=pl.Series([1, 0], dtype=pl.UInt32))
    assert_frame_equal(result, expected)


def test_label_encode_inverse_context():
    df = DataFrame({"cat": ["apple", "banana"]})

    input_df = df.clone()

    pipe = Pipeline()

    inverse_map = {}

    with pipe.label_encode("cat", inverse_mapping=inverse_map) as p:
        transformed = p.fit_transform(input_df)

        assert transformed["cat"].dtype == pl.UInt32
        assert transformed["cat"].to_list() == [0, 1]

    pipe2 = Pipeline()
    with pipe2.label_encode("cat", inverse_mapping={"cat_restored": "cat"}) as p2:
        encoded = p2.fit_transform(df)
        assert encoded["cat"].dtype == pl.UInt32

    pipe3 = Pipeline()
    with pipe3.label_encode("cat", inverse_mapping={"cat": "cat"}):
        pass

    result = pipe3.fit_transform(df)

    assert "cat" in result.columns
    assert result["cat"].dtype == pl.String
    assert result["cat"].to_list() == ["apple", "banana"]


def test_label_encode_not_fitted():
    df = DataFrame({"cat": ["a"]})
    pipeline = Pipeline().label_encode("cat")
    with pytest.raises(NotFittedError):
        pipeline.transform(df)
