from __future__ import annotations

import inspect
import sys
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Mapping,
    Sequence,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import polars as pl


from const import END_INSERTION_MARKER, START_INSERTION_MARKER
from polars_ml import Pipeline
from utils import insert_between_markers



def insert_pipeline_tests():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    target_file = PROJECT_ROOT / "tests" / "polars_ml" / "pipeline" / "test_pipeline.py"

    codes = render_tests()
    insert_between_markers(
        target_file,
        "".join(codes),
        START_INSERTION_MARKER.format(prefix="", suffix=" IN Pipeline Tests ---"),
        END_INSERTION_MARKER.format(prefix="", suffix=" IN Pipeline Tests ---"),
    )


if __name__ == "__main__":
    insert_pipeline_tests()
