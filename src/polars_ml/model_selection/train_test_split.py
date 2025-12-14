import uuid

import polars as pl
from polars import DataFrame, Series


def train_test_split(
    data: DataFrame,
    test_size: float,
    *,
    shuffle: bool = False,
    seed: int | None = None,
    stratify: str | None = None,
) -> tuple[Series, Series]:
    index_name = train_test_split.__name__ + "_" + uuid.uuid4().hex
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

    if shuffle:
        shuffle_expr = pl.col("is_train").shuffle(seed)
        if stratify is not None:
            shuffle_expr = shuffle_expr.over(stratify)
        data = data.with_columns(shuffle_expr)

    return (
        data.filter(pl.col("is_train"))[index_name],
        data.filter(~pl.col("is_train"))[index_name],
    )
