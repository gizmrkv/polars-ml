from datetime import datetime

import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal

from polars_ml import Pipeline


@pytest.mark.parametrize(
    "data",
    [
        DataFrame(
            {
                "f0": ["a", "a", "b", "b", "c"],
                "f1": [0.1, 0.2, 0.3, 0.4, 0.5],
                "f2": [True, True, False, True, False],
            }
        )
    ],
)
def test_pipeline_group_by(data: DataFrame):
    for method, args, kwargs in [
        ("agg", [pl.col("f1").mean()], {}),
        ("all", [], {}),
        ("first", [], {}),
        ("head", [2], {}),
        ("last", [], {}),
        ("len", ["len"], {}),
        ("map_groups", [lambda df: df.sample(2, seed=42, with_replacement=True)], {}),  # type: ignore
        ("max", [], {}),
        ("mean", [], {}),
        ("median", [], {}),
        ("min", [], {}),
        ("n_unique", [], {}),
        ("quantile", [0.5], {}),
        ("sum", [], {}),
        ("tail", [2], {}),
    ]:
        pp = getattr(Pipeline().group_by("f0", maintain_order=True), method)(
            *args, **kwargs
        )
        out = pp.transform(data)
        exp = getattr(data.group_by("f0", maintain_order=True), method)(*args, **kwargs)
        assert_frame_equal(out, exp)


@pytest.mark.parametrize(
    "data",
    [
        DataFrame(
            {
                "time": pl.datetime_range(
                    start=datetime(2021, 12, 16),
                    end=datetime(2021, 12, 16, 3),
                    interval="30m",
                    eager=True,
                ),
                "n": range(7),
            }
        )
    ],
)
def test_pipeline_group_by_dynamic(data: DataFrame):
    pp = (
        Pipeline().group_by_dynamic("time", every="1h", closed="right").agg(pl.col("n"))
    )
    out = pp.transform(data)
    exp = data.group_by_dynamic("time", every="1h", closed="right").agg(pl.col("n"))
    assert_frame_equal(out, exp)


@pytest.mark.parametrize(
    "data",
    [
        DataFrame(
            {
                "dt": [
                    "2020-01-01 13:45:48",
                    "2020-01-01 16:42:13",
                    "2020-01-01 16:45:09",
                    "2020-01-02 18:12:48",
                    "2020-01-03 19:45:32",
                    "2020-01-08 23:16:43",
                ],
                "a": [3, 7, 5, 9, 2, 1],
            }
        ).with_columns(pl.col("dt").str.strptime(pl.Datetime).set_sorted())
    ],
)
def test_pipeline_group_by_rolling(data: DataFrame):
    pp = (
        Pipeline()
        .rolling(index_column="dt", period="2d")
        .agg(
            pl.sum("a").alias("sum_a"),
            pl.min("a").alias("min_a"),
            pl.max("a").alias("max_a"),
        )
    )
    out = pp.transform(data)
    exp = data.rolling(index_column="dt", period="2d").agg(
        pl.sum("a").alias("sum_a"),
        pl.min("a").alias("min_a"),
        pl.max("a").alias("max_a"),
    )
    assert_frame_equal(out, exp)
