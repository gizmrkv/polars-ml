from datetime import datetime

import polars as pl
import pytest
from polars import DataFrame, Series
from polars.testing import assert_frame_equal

from polars_ml import Pipeline
from polars_ml.pipeline.testing import assert_component_valid


def test_pipeline_join_small(test_data_small: DataFrame):
    other = DataFrame({"s0": ["a", "b"], "other": [1, 2]})
    assert_component_valid(
        Pipeline().join(other, on="s0"),
        test_data_small,
        test_data_small.join(other, on="s0"),
    )
    assert_component_valid(
        Pipeline().join(Pipeline().select("s0").with_row_index(), on="s0"),
        test_data_small,
        test_data_small.join(test_data_small.select("s0").with_row_index(), on="s0"),
    )


@pytest.mark.parametrize(
    ["data", "other"],
    [
        (
            DataFrame(
                {
                    "time": [
                        datetime(2020, 1, 1, 9, 0),
                        datetime(2020, 1, 1, 9, 1),
                        datetime(2020, 1, 1, 9, 2),
                    ],
                    "value": [1, 2, 3],
                }
            ),
            DataFrame(
                {
                    "time": [
                        datetime(2020, 1, 1, 9, 0, 30),
                        datetime(2020, 1, 1, 9, 1, 30),
                    ],
                    "value": [10, 20],
                }
            ),
        ),
    ],
)
def test_pipeline_join_asof(data: DataFrame, other: DataFrame):
    pp = Pipeline().join_asof(other, on="time")
    out = pp.transform(data)
    exp = data.join_asof(other, on="time")
    assert_frame_equal(out, exp)

    other_pp = Pipeline().select(["time", "value"]).sort("time")
    pp = Pipeline().join_asof(other_pp, on="time")
    out = pp.transform(data)
    exp = data.join_asof(other_pp.transform(data), on="time")
    assert_frame_equal(out, exp)


@pytest.mark.parametrize(
    ["data", "other"],
    [
        (
            DataFrame(
                {
                    "id": [100, 101, 102],
                    "dur": [120, 140, 160],
                    "rev": [12, 14, 16],
                    "cores": [2, 8, 4],
                }
            ),
            DataFrame(
                {
                    "t_id": [404, 498, 676, 742],
                    "time": [90, 130, 150, 170],
                    "cost": [9, 13, 15, 16],
                    "cores": [4, 2, 1, 4],
                }
            ),
        )
    ],
)
def test_pipeline_join_where(data: DataFrame, other: DataFrame):
    predicates = [
        pl.col("dur") < pl.col("time"),
        pl.col("rev") < pl.col("cost"),
    ]

    pp = Pipeline().join_where(other, *predicates)
    out = pp.transform(data)
    exp = data.join_where(other, *predicates)
    assert_frame_equal(out, exp)

    other_pp = Pipeline().with_columns(
        pl.col("dur").alias("time"), pl.col("rev").alias("cost")
    )
    pp = Pipeline().join_where(other_pp, *predicates)
    out = pp.transform(data)
    exp = data.join_where(other_pp.transform(data), *predicates)
    assert_frame_equal(out, exp)


@pytest.mark.parametrize(
    ["data", "other"],
    [
        (
            DataFrame(
                {
                    "name": ["steve", "elise", "bob"],
                    "age": [42, 44, 18],
                }
            ).sort("age"),
            DataFrame(
                {
                    "name": ["anna", "megan", "steve", "thomas"],
                    "age": [21, 33, 42, 20],
                }
            ).sort("age"),
        )
    ],
)
def test_pipeline_merge_sorted(data: DataFrame, other: DataFrame):
    pp = Pipeline().merge_sorted(other, key="age")
    out = pp.transform(data)
    exp = data.merge_sorted(other, key="age")
    assert_frame_equal(out, exp)

    other_pp = Pipeline().sort("age")
    pp = Pipeline().merge_sorted(other_pp, key="age")
    out = pp.transform(data)
    exp = data.merge_sorted(other_pp.transform(data), key="age")
    assert_frame_equal(out, exp)


@pytest.mark.parametrize(
    ["data", "other"],
    [
        (
            DataFrame(
                {
                    "A": [1, 2, 3, 4],
                    "B": [400, 500, 600, 700],
                }
            ),
            DataFrame(
                {
                    "B": [-66, None, -99],
                    "C": [5, 3, 1],
                }
            ),
        )
    ],
)
def test_pipeline_update(data: DataFrame, other: DataFrame):
    pp = Pipeline().update(other)
    out = pp.transform(data)
    exp = data.update(other)
    assert_frame_equal(out, exp)

    other_pp = Pipeline().select("B")
    pp = Pipeline().update(other_pp)
    out = pp.transform(data)
    exp = data.update(other_pp.transform(data))
    assert_frame_equal(out, exp)


@pytest.mark.parametrize(
    ["data", "other"],
    [
        (
            DataFrame(
                {
                    "foo": [1, 2],
                    "bar": [6, 7],
                    "ham": ["a", "b"],
                }
            ),
            DataFrame(
                {
                    "foo": [3, 4],
                    "bar": [8, 9],
                    "ham": ["c", "d"],
                }
            ),
        )
    ],
)
def test_pipeline_vstack(data: DataFrame, other: DataFrame):
    pp = Pipeline().vstack(other)
    out = pp.transform(data)
    exp = data.vstack(other)
    assert_frame_equal(out, exp)

    other_pp = Pipeline().echo()
    pp = Pipeline().vstack(other_pp)
    out = pp.transform(data)
    exp = data.vstack(other_pp.transform(data))
    assert_frame_equal(out, exp)


@pytest.mark.parametrize(
    ["data", "other"],
    [
        (
            DataFrame(
                {
                    "foo": [1, 2],
                    "bar": [6, 7],
                    "ham": ["a", "b"],
                }
            ),
            DataFrame(
                {
                    "foo": [3, 4],
                    "bar": [8, 9],
                    "ham": ["c", "d"],
                }
            ),
        )
    ],
)
def test_pipeline_concat(data: DataFrame, other: DataFrame):
    pp = Pipeline().concat(data, other)
    out = pp.transform(data)
    exp = pl.concat([data, other])
    assert_frame_equal(out, exp)

    other_pp = Pipeline().clone()
    pp = Pipeline().concat(other_pp, other)
    out = pp.transform(data)
    exp = pl.concat([other_pp.transform(data), other])
    assert_frame_equal(out, exp)
