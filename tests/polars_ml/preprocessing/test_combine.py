from __future__ import annotations

import pickle

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_ml.preprocessing.combine import Combine


def test_combine_fit_transform():
    df = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        }
    )

    # Combinations of 2 from (a, b, c) -> (a,b), (a,c), (b,c)
    transformer = Combine(columns=["a", "b", "c"], n=2)
    transformer.fit(df)

    assert len(transformer.combinations_) == 3
    assert ("a", "b") in transformer.combinations_
    assert ("a", "c") in transformer.combinations_
    assert ("b", "c") in transformer.combinations_

    result = transformer.transform(df)

    expected = df.with_columns(
        [
            pl.struct(["a", "b"]).alias("comb_a_b"),
            pl.struct(["a", "c"]).alias("comb_a_c"),
            pl.struct(["b", "c"]).alias("comb_b_c"),
        ]
    )

    assert_frame_equal(result, expected)


def test_combine_fit_transform_n3():
    df = pl.DataFrame(
        {
            "a": [1],
            "b": [4],
            "c": [7],
        }
    )

    # Combinations of 3 from (a, b, c) -> (a,b,c)
    transformer = Combine(columns=["a", "b", "c"], n=3)
    transformer.fit(df)

    assert len(transformer.combinations_) == 1
    assert transformer.combinations_[0] == ("a", "b", "c")

    result = transformer.transform(df)
    expected = df.with_columns(pl.struct(["a", "b", "c"]).alias("comb_a_b_c"))

    assert_frame_equal(result, expected)


def test_combine_fit_custom_params():
    df = pl.DataFrame({"x": [1], "y": [2]})
    transformer = Combine(["x", "y"], 2, delimiter="_", prefix="group_")

    transformer.fit(df)
    result = transformer.transform(df)

    expected = df.with_columns(pl.struct(["x", "y"]).alias("group_x_y"))
    assert_frame_equal(result, expected)
