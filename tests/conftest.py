import polars as pl
import pytest
from polars import DataFrame, Series


@pytest.fixture
def test_data_small():
    return DataFrame(
        [
            Series("i0", [-1, 0, 1, 2], dtype=pl.Int32),
            Series("i1", [0, 0, 0, 0], dtype=pl.Int32),
            Series("i2", [1, None, 3, 4], dtype=pl.Int32),
            Series("u0", [0, 1, 2, 3], dtype=pl.UInt32),
            Series("u1", [0, 0, 0, 0], dtype=pl.UInt32),
            Series("u2", [1, None, 3, 4], dtype=pl.UInt32),
            Series("f0", [-0.1, 0.0, 0.1, 0.2], dtype=pl.Float64),
            Series("f1", [0.0, 0.0, 0.0, 0.0], dtype=pl.Float64),
            Series("f2", [0.1, None, 0.3, float("nan")], dtype=pl.Float64),
            Series("s0", ["a", "b", "c", "c"], dtype=pl.String),
            Series("s1", ["a", "a", "a", "a"], dtype=pl.String),
            Series("s2", ["a", None, "c", "c"], dtype=pl.String),
            Series("b0", [True, False, True, False], dtype=pl.Boolean),
            Series("b1", [True, False, False, False], dtype=pl.Boolean),
            Series("b2", [True, None, False, False], dtype=pl.Boolean),
        ]
    )
