import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_ml.preprocessing.join_agg import JoinAgg


def test_join_agg_basic():
    df = pl.DataFrame({"group": ["A", "A", "B", "B", "C"], "value": [1, 2, 3, 4, 5]})

    # Calculate mean per group
    transformer = JoinAgg(
        "group", pl.col("value").mean().alias("mean_value"), how="left"
    )

    transformer.fit(df)
    result = transformer.transform(df)

    expected = df.with_columns(pl.Series("mean_value", [1.5, 1.5, 3.5, 3.5, 5.0]))

    assert_frame_equal(result, expected)


def test_join_agg_multiple_keys():
    df = pl.DataFrame(
        {"k1": ["A", "A", "B", "B"], "k2": [1, 1, 2, 2], "val": [10, 20, 30, 40]}
    )

    transformer = JoinAgg(["k1", "k2"], pl.col("val").sum().alias("sum_val"))

    result = transformer.fit_transform(df)

    expected = df.with_columns(pl.Series("sum_val", [30, 30, 70, 70]))

    assert_frame_equal(result, expected)


def test_join_agg_unseen_category():
    train_df = pl.DataFrame({"group": ["A", "B"], "val": [1, 2]})

    test_df = pl.DataFrame(
        {
            "group": ["A", "C"],
            "val": [1, 5],
        }
    )

    transformer = JoinAgg("group", pl.col("val").max().alias("max_val"))

    transformer.fit(train_df)
    result = transformer.transform(test_df)

    expected = test_df.with_columns(pl.Series("max_val", [1, None]))

    assert_frame_equal(result, expected)


def test_join_agg_suffix_collision():
    df = pl.DataFrame({"group": ["A", "A"], "val": [1, 2]})

    transformer = JoinAgg("group", pl.col("val").mean().alias("val"))

    transformer.fit(df)
    result = transformer.transform(df)

    assert "val_agg" in result.columns
    assert result["val_agg"][0] == 1.5
