import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_ml.preprocessing.agg_join import AggJoin


def test_agg_join_basic():
    df = pl.DataFrame({"group": ["A", "A", "B", "B", "C"], "value": [1, 2, 3, 4, 5]})

    # Calculate mean per group
    transformer = AggJoin(
        by="group", aggs=pl.col("value").mean().alias("mean_value"), how="left"
    )

    transformer.fit(df)
    result = transformer.transform(df)

    expected = df.with_columns(pl.Series("mean_value", [1.5, 1.5, 3.5, 3.5, 5.0]))

    assert_frame_equal(result, expected)


def test_agg_join_multiple_keys():
    df = pl.DataFrame(
        {"k1": ["A", "A", "B", "B"], "k2": [1, 1, 2, 2], "val": [10, 20, 30, 40]}
    )

    transformer = AggJoin(by=["k1", "k2"], aggs=pl.col("val").sum().alias("sum_val"))

    result = transformer.fit_transform(df)

    expected = df.with_columns(pl.Series("sum_val", [30, 30, 70, 70]))

    assert_frame_equal(result, expected)


def test_agg_join_unseen_category():
    train_df = pl.DataFrame({"group": ["A", "B"], "val": [1, 2]})

    test_df = pl.DataFrame(
        {
            "group": ["A", "C"],  # C is unseen
            "val": [1, 5],
        }
    )

    transformer = AggJoin(by="group", aggs=pl.col("val").max().alias("max_val"))

    transformer.fit(train_df)
    result = transformer.transform(test_df)

    # For "A", max is 1. For "C", it should be null (left join)
    expected = test_df.with_columns(pl.Series("max_val", [1, None]))

    assert_frame_equal(result, expected)


def test_agg_join_suffix_collision():
    df = pl.DataFrame({"group": ["A", "A"], "val": [1, 2]})

    # If the agg result has same name as existing column (besides join keys),
    # suffix should be applied if it's not the join key?
    # Actually join keys are merged.
    # If I aggregate to a name that already exists in df but is NOT a join key:

    transformer = AggJoin(
        by="group",
        aggs=pl.col("val").mean().alias("val"),  # Collision with input "val"
    )

    transformer.fit(df)
    result = transformer.transform(df)

    # Should have "val" (original) and "val_agg" (joined)
    assert "val_agg" in result.columns
    assert result["val_agg"][0] == 1.5
