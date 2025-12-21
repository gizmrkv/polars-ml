from __future__ import annotations

from datetime import datetime

import polars as pl
from polars import DataFrame
import pytest
from polars_ml import Pipeline


@pytest.fixture
def df() -> DataFrame:
    return pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
            "c": ["x", "y", "z", "x", "y"],
            "d": pl.date_range(
                datetime(2020, 1, 1), datetime(2020, 1, 5), "1d", eager=True
            ),
        }
    )


@pytest.fixture
def other_df() -> DataFrame:
    return pl.DataFrame({"a": [1, 2, 3, 4, 5], "e": [0.1, 0.2, 0.3, 0.4, 0.5]})


# --- START INSERTION MARKER IN Pipeline Tests ---
# --- END INSERTION MARKER IN Pipeline Tests ---
