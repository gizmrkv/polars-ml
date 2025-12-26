from pathlib import Path

import kagglehub
import polars as pl

from polars_ml import Pipeline
from polars_ml.model_selection import train_test_split


def main():
    path = Path(kagglehub.dataset_download("uciml/adult-census-income"))

    data = pl.read_csv(path / "*.csv")

    train_idx, test_idx = train_test_split(data, 0.2, shuffle=True, seed=42)
    train_data = data.select(pl.all().gather(train_idx))
    test_data = data.select(pl.all().gather(test_idx))

    train_idx, valid_idx = train_test_split(train_data, 0.4, shuffle=True, seed=42)
    valid_data = train_data.select(pl.all().gather(valid_idx))
    train_data = train_data.select(pl.all().gather(train_idx))

    target_column = "income"
    pp = (
        Pipeline()
        .label_encode(
            "workclass",
            "education",
            "marital.status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native.country",
        )
        .with_columns(pl.col(target_column).eq(pl.lit(">50K")))
        .gbdt.lightgbm(
            {"objective": "binary", "metric": "binary_logloss"},
            target_column,
        )
    )

    pp.fit_transform(train_data, valid=valid_data)
    test_pred = pp.transform(test_data)

    evaluate = Pipeline().metrics.binary_classification(target_column, "prediction")
    print(evaluate.fit_transform(test_pred))


if __name__ == "__main__":
    main()
