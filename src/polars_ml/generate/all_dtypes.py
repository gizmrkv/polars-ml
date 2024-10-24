import datetime
import random
from typing import List

import numpy as np
import polars as pl
from polars import DataFrame, Series


def generate_all_dtypes(
    height: int, *, null_rate: float = 0.0, seed: int | None = None
) -> DataFrame:
    if seed is not None:
        random_state = random.Random(seed)
        random_state_np = np.random.RandomState(seed)
    else:
        random_state = random
        random_state_np = np.random

    data: List[Series] = []
    for float_dtype in [pl.Float32, pl.Float64]:
        data.append(
            Series(
                float_dtype.__name__,
                [
                    None if random_state.random() < null_rate else v
                    for v in random_state_np.normal(size=height)
                ],
                dtype=float_dtype,
            )
        )

    for int_dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
        data.append(
            Series(
                int_dtype.__name__,
                [
                    None if random_state.random() < null_rate else v
                    for v in random_state_np.randint(-10, 10, size=height)
                ],
                dtype=int_dtype,
            )
        )

    for uint_dtype in [pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
        data.append(
            Series(
                uint_dtype.__name__,
                [
                    None if random_state.random() < null_rate else v
                    for v in random_state_np.randint(0, 10, size=height)
                ],
                dtype=uint_dtype,
            )
        )

    for str_dtype in [pl.String, pl.Categorical]:
        data.append(
            Series(
                str_dtype.__name__,
                [
                    None if random_state.random() < null_rate else f"str_{v}"
                    for v in random_state_np.randint(0, 10, size=height)
                ],
                dtype=str_dtype,
            )
        )

    data.append(
        Series(
            pl.Boolean.__name__,
            [
                None
                if random_state.random() < null_rate
                else random_state.choice([True, False])
                for _ in range(height)
            ],
            dtype=pl.Boolean,
        )
    )

    data.append(
        Series(
            pl.Date.__name__,
            [
                None
                if random_state.random() < null_rate
                else datetime.date.fromisoformat(
                    f"{random_state_np.randint(2000, 2020)}-"
                    + f"{random_state_np.randint(1, 12):02d}-"
                    + f"{random_state_np.randint(1, 28):02d}"
                )
                for _ in range(height)
            ],
            dtype=pl.Date,
        )
    )

    data.append(
        Series(
            pl.Datetime.__name__,
            [
                None
                if random_state.random() < null_rate
                else datetime.datetime.fromisoformat(
                    f"{random_state_np.randint(2000, 2020)}-"
                    + f"{random_state_np.randint(1, 12):02d}-"
                    + f"{random_state_np.randint(1, 28):02d} "
                    + f"{random_state_np.randint(0, 24):02d}:"
                    + f"{random_state_np.randint(0, 60):02d}:"
                    + f"{random_state_np.randint(0, 60):02d}"
                )
                for _ in range(height)
            ],
            dtype=pl.Datetime,
        )
    )

    data.append(
        Series(
            pl.Array.__name__,
            [
                None
                if random_state.random() < null_rate
                else [random_state.random() for _ in range(5)]
                for _ in range(height)
            ],
            dtype=pl.Array(pl.Float32, 5),
        )
    )

    data.append(
        Series(
            pl.List.__name__,
            [
                None
                if random_state.random() < null_rate
                else [random_state.random() for _ in range(random_state.randint(0, 10))]
                for _ in range(height)
            ],
            dtype=pl.List,
        )
    )

    data.append(
        Series(
            pl.Struct.__name__,
            [
                None
                if random_state.random() < null_rate
                else {
                    "a": random_state.random(),
                    "b": random_state.randint(0, 10),
                }
                for _ in range(height)
            ],
            dtype=pl.Struct,
        )
    )

    return DataFrame(data)
