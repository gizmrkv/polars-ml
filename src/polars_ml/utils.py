from typing import Iterable

import polars as pl
from polars import DataFrame


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
