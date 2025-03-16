import numpy as np
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


@pytest.fixture
def test_data_medium():
    np.random.seed(42)
    size = 1000
    return DataFrame(
        [
            Series("i0", np.random.randint(-100, 100, size), dtype=pl.Int32),
            Series("i1", np.random.randint(-100, 100, size), dtype=pl.Int32),
            Series("i2", np.random.randint(-100, 100, size), dtype=pl.Int32),
            Series("u0", np.random.randint(0, 100, size), dtype=pl.UInt32),
            Series("u1", np.random.randint(0, 100, size), dtype=pl.UInt32),
            Series("u2", np.random.randint(0, 100, size), dtype=pl.UInt32),
            Series("f0", np.random.randn(size), dtype=pl.Float64),
            Series("f1", np.random.randn(size), dtype=pl.Float64),
            Series("f2", np.random.randn(size), dtype=pl.Float64),
            Series("s0", np.random.choice(["a", "b", "c"], size), dtype=pl.String),
            Series("s1", np.random.choice(["a", "b", "c"], size), dtype=pl.String),
            Series("s2", (["a", "b", "c"] * (size // 3 + 3))[:size], dtype=pl.String),
            Series("b0", np.random.choice([True, False], size), dtype=pl.Boolean),
            Series("b1", np.random.choice([True, False], size), dtype=pl.Boolean),
            Series("b2", np.random.choice([True, False], size), dtype=pl.Boolean),
        ]
    )
