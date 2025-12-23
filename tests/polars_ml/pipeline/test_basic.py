import polars as pl
import pytest
from polars import DataFrame
from polars.testing import assert_frame_equal

from polars_ml import Transformer
from polars_ml.pipeline.basic import Apply, Concat, Const, Echo, Parrot, Side, ToDummies


@pytest.fixture
def sample_df() -> DataFrame:
    return pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def test_apply(sample_df: DataFrame):
    t = Apply(lambda df: df.select(pl.col("a") * 2))
    output = t.transform(sample_df)
    expected = sample_df.select(pl.col("a") * 2)
    assert_frame_equal(output, expected)


def test_echo(sample_df: DataFrame):
    t = Echo()
    output = t.transform(sample_df)
    assert_frame_equal(output, sample_df)


def test_const(sample_df: DataFrame):
    const_df = DataFrame({"x": [1]})
    t = Const(const_df)
    output = t.transform(sample_df)
    assert_frame_equal(output, const_df)


def test_parrot(sample_df: DataFrame):
    t = Parrot()
    t.fit(sample_df)
    output = t.transform(sample_df)
    assert_frame_equal(output, sample_df)


def test_side(sample_df: DataFrame):
    class MockTransformer(Transformer):
        def __init__(self):
            self.called = False

        def transform(self, data: DataFrame) -> DataFrame:
            self.called = True
            return data

    mock = MockTransformer()
    t = Side(mock)

    output = t.transform(sample_df)
    assert_frame_equal(output, sample_df)


def test_concat_vertical(sample_df: DataFrame):
    t = Concat([Echo(), Echo()], how="vertical")

    output = t.transform(sample_df)
    expected = pl.concat([sample_df, sample_df], how="vertical")
    assert_frame_equal(output, expected)


def test_concat_horizontal(sample_df: DataFrame):
    t = Concat(
        [Apply(lambda df: df.select("a")), Apply(lambda df: df.select("b"))],
        how="horizontal",
    )

    output = t.transform(sample_df)
    expected = pl.concat(
        [sample_df.select("a"), sample_df.select("b")], how="horizontal"
    )
    assert_frame_equal(output, expected)


def test_to_dummies(sample_df: DataFrame):
    df = DataFrame({"cat": ["A", "B", "A"]})
    t = ToDummies(columns=["cat"])
    t.fit(df)
    output = t.transform(df)
    assert "cat_A" in output.columns
    assert "cat_B" in output.columns
    assert output["cat_A"].to_list() == [1, 0, 1]
    assert output["cat_B"].to_list() == [0, 1, 0]
