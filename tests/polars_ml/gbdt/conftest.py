import tempfile
from collections.abc import Generator
from pathlib import Path

import numpy as np
import polars as pl
import pytest


def get_mock_data() -> pl.DataFrame:
    np.random.seed(42)
    data = pl.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "target": np.random.randint(0, 2, 100),
        }
    )
    return data


@pytest.fixture
def catboost_tmpdir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
