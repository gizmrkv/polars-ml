import uuid

import polars as pl
from polars import DataFrame, Series


def train_test_split(
    data: DataFrame,
    test_size: float,
    *,
    stratify: str | None = None,
    shuffle: bool = True,
    seed: int | None = None,
) -> tuple[Series, Series]:
    index_name = uuid.uuid4().hex
    if stratify is None:
        data = data.with_row_index(index_name).select(
            index_name,
            pl.col(index_name).alias("is_train") > data.height * test_size,
        )
    else:
        data = data.with_row_index(index_name).select(
            index_name,
            stratify or pl.lit(0),
            pl.cum_count(stratify).over(stratify).alias("is_train")
            > pl.len().over(stratify) * test_size,
        )

    data = data.with_columns(
        pl.col("is_train").shuffle(seed).over(stratify) if shuffle else pl.all()
    )
    return (
        data.filter(pl.col("is_train"))[index_name],
        data.filter(~pl.col("is_train"))[index_name],
    )
