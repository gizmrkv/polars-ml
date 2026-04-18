import polars as pl
from polars.testing import assert_frame_equal

from polars_ml.pipeline import Pipeline
from polars_ml.pipeline.group_by import GroupByGetAttr


def test_group_by_get_attr_basic() -> None:
    df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})

    step = GroupByGetAttr("group_by", "agg", ("g",), {}, pl.col("v").sum())
    transformed = step.transform(df).sort("g")

    expected = df.group_by("g").agg(pl.col("v").sum()).sort("g")
    assert_frame_equal(transformed, expected)


def test_pipeline_group_by_namespace() -> None:
    df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})

    pipe = Pipeline().group_by("g").sum()
    transformed = pipe.transform(df).sort("g")

    expected = df.group_by("g").sum().sort("g")
    assert_frame_equal(transformed, expected)


def test_pipeline_group_by_agg() -> None:
    df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})

    pipe = Pipeline().group_by("g").agg(pl.col("v").max().alias("v_max"))
    transformed = pipe.transform(df).sort("g")

    expected = df.group_by("g").agg(pl.col("v").max().alias("v_max")).sort("g")
    assert_frame_equal(transformed, expected)


def test_pipeline_group_by_methods() -> None:
    df = pl.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})

    methods = ["len", "first", "last", "mean", "min", "max"]
    for method in methods:
        pipe = getattr(Pipeline().group_by("g"), method)()
        transformed = pipe.transform(df).sort("g")
        expected = getattr(df.group_by("g"), method)().sort("g")
        assert_frame_equal(transformed, expected)


def test_pipeline_group_by_dynamic() -> None:
    df = pl.DataFrame(
        {
            "time": pl.datetime_range(
                start=pl.datetime(2023, 1, 1),
                end=pl.datetime(2023, 1, 2),
                interval="1h",
                eager=True,
            ),
            "v": range(25),
        }
    )

    pipe = Pipeline().group_by_dynamic("time", every="1d").agg(pl.col("v").sum())
    transformed = pipe.transform(df)

    expected = df.group_by_dynamic("time", every="1d").agg(pl.col("v").sum())
    assert_frame_equal(transformed, expected)


def test_pipeline_rolling() -> None:
    df = pl.DataFrame(
        {
            "time": pl.datetime_range(
                start=pl.datetime(2023, 1, 1),
                end=pl.datetime(2023, 1, 2),
                interval="1h",
                eager=True,
            ),
            "v": range(25),
        }
    )

    pipe = Pipeline().rolling("time", period="3h").agg(pl.col("v").sum())
    transformed = pipe.transform(df)

    expected = df.rolling("time", period="3h").agg(pl.col("v").sum())
    assert_frame_equal(transformed, expected)
