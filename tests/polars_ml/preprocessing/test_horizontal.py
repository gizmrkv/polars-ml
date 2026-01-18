import polars as pl
import polars.testing as pt

from polars_ml import Pipeline


def test_horizontal() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )

    pipeline = Pipeline().horizontal.sum("a", "b", value_name="total")

    result = pipeline.fit_transform(df)

    expected = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "total": [5, 7, 9],
        }
    )

    pt.assert_frame_equal(result, expected)


def test_horizontal_mean() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [3, 4, 5],
        }
    )

    pipeline = Pipeline().horizontal.mean(pl.col("a"), pl.col("b"), value_name="avg")

    result = pipeline.fit_transform(df)

    expected = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [3, 4, 5],
            "avg": [2.0, 3.0, 4.0],
        }
    )

    pt.assert_frame_equal(result, expected)


def test_horizontal_argmax() -> None:
    df = pl.DataFrame(
        {
            "v1": [1, 10, 5],
            "v2": [2, 1, 6],
        }
    )

    pipeline = Pipeline().horizontal.arg_max("v1", "v2", value_name="max_col")

    result = pipeline.fit_transform(df)

    # HorizontalArgMax returns a list of column names
    expected = pl.DataFrame(
        {
            "v1": [1, 10, 5],
            "v2": [2, 1, 6],
            "max_col": [["v2"], ["v1"], ["v2"]],
        }
    )

    pl.testing.assert_frame_equal(result, expected)
