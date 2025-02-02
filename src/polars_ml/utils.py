import random
from typing import Iterable, Iterator, Literal, TypeVar

import polars as pl
from polars import DataFrame

T = TypeVar("T")


def get_country_codes() -> DataFrame:
    import pycountry

    columns = ["name", "alpha_2", "alpha_3", "numeric", "flag"]
    return DataFrame(
        {
            column: [getattr(country, column) for country in pycountry.countries]
            for column in columns
        }
    )


def get_country_holidays(
    countries: str | Iterable[str], years: int | Iterable[int]
) -> DataFrame:
    import holidays

    return pl.concat(
        [
            pl.DataFrame({"date": dates, "holiday": names, "country": country})
            for country in countries
            for dates, names in [
                zip(*holidays.country_holidays(country, years=years).items())
            ]
        ]
    )


def incremental_sampling(n_rows: int, n_blocks: int) -> Iterator[list[int]]:
    if n_rows < 2**n_blocks:
        raise ValueError("n_rows must be less than or equal to 2 ** n_subsamples")

    indexes = list(range(n_rows))
    random.shuffle(indexes)
    for i in range(n_blocks - 1, -1, -1):
        yield indexes[: n_rows >> i]


def deduplicate_scores(
    scores: dict[T, float],
    *,
    tolerance: float = 1e-20,
    direction: Literal["maximize", "minimize"] = "maximize",
) -> dict[T, float]:
    name_scores = sorted(
        scores.items(), key=lambda x: x[1] if direction == "maximize" else -x[1]
    )
    deduplicated_scores = {}
    prev_score = name_scores[0][1] + 2 * tolerance
    for name, score in name_scores:
        if score < prev_score - tolerance:
            deduplicated_scores[name] = score
            prev_score = score

    return deduplicated_scores
