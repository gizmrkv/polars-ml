import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_ml.exceptions import NotFittedError
from polars_ml.pipeline import Pipeline
from polars_ml.pipeline.label_encode import LabelEncode


def test_label_encode_fit_transform() -> None:
    df = pl.DataFrame(
        {
            "a": ["x", "y", "x", "z"],
            "b": ["apple", "banana", "apple", "apple"],
        }
    )

    encoder = LabelEncode("a", "b")
    encoded = encoder.fit_transform(df)

    expected = pl.DataFrame(
        {
            "a": [0, 1, 0, 2],
            "b": [0, 1, 0, 0],
        },
        schema={"a": pl.UInt32, "b": pl.UInt32},
    )

    assert_frame_equal(encoded, expected)


def test_label_encode_with_orders() -> None:
    df = pl.DataFrame(
        {
            "a": ["x", "y", "x", "z"],
        }
    )

    encoder = LabelEncode("a", orders={"a": ["y", "z", "x"]})
    encoded = encoder.fit_transform(df)

    expected = pl.DataFrame(
        {
            "a": [2, 0, 2, 1],
        },
        schema={"a": pl.UInt32},
    )

    assert_frame_equal(encoded, expected)


def test_label_encode_maintain_order() -> None:
    df = pl.DataFrame(
        {
            "a": ["x", "y", "x", "z"],
        }
    )
    encoder = LabelEncode("a", maintain_order=False)
    encoded = encoder.fit_transform(df)

    assert encoded.schema["a"] == pl.UInt32
    assert encoded["a"].n_unique() == 3


def test_label_encode_missing_column() -> None:
    df_fit = pl.DataFrame({"a": ["x", "y"]})
    encoder = LabelEncode("a").fit(df_fit)

    df_transform = pl.DataFrame({"b": ["apple", "banana"]})
    # transform for LazyTransformer takes LazyFrame
    encoded = encoder.transform(df_transform.lazy()).collect()

    assert_frame_equal(encoded, df_transform)


def test_label_encode_not_fitted() -> None:
    encoder = LabelEncode("a")
    df = pl.DataFrame({"a": ["x", "y"]})

    with pytest.raises(NotFittedError):
        encoder.transform(df.lazy())

    with pytest.raises(NotFittedError):
        _ = encoder.mappings


def test_label_encode_mappings() -> None:
    df = pl.DataFrame({"a": ["x", "y"]})
    encoder = LabelEncode("a").fit(df)

    mapping = encoder.mappings["a"]
    expected = pl.DataFrame(
        {
            "key": ["x", "y"],
            "value": [0, 1],
        },
        schema={"key": pl.String, "value": pl.UInt32},
    )

    assert_frame_equal(mapping, expected)


def test_label_encode_pipeline() -> None:
    df = pl.DataFrame(
        {
            "a": ["x", "y", "x", "z"],
            "b": ["apple", "banana", "apple", "apple"],
            "c": [1, 2, 3, 4],
        }
    )

    pipe = Pipeline().label_encode("a", "b")
    encoded = pipe.fit_transform(df)

    expected = pl.DataFrame(
        {
            "a": [0, 1, 0, 2],
            "b": [0, 1, 0, 0],
            "c": [1, 2, 3, 4],
        },
        schema={"a": pl.UInt32, "b": pl.UInt32, "c": pl.Int64},
    )

    assert_frame_equal(encoded, expected)
