from pathlib import Path

import kagglehub
import polars as pl

from polars_ml import Pipeline
from polars_ml.model_selection import train_test_split


def main():
    path = Path(kagglehub.dataset_download("uciml/adult-census-income"))

    data = pl.read_csv(path / "*.csv")

    train_valid_idx, test_idx = train_test_split(data, 0.2, shuffle=True, seed=42)
    train_valid_data = data.select(pl.all().gather(train_valid_idx))
    test_data = data.select(pl.all().gather(test_idx))

    train_idx, valid_idx = train_test_split(
        train_valid_data, 0.4, shuffle=True, seed=42
    )
    valid_data = train_valid_data.select(pl.all().gather(valid_idx))
    train_data = train_valid_data.select(pl.all().gather(train_idx))

    target_column = "income"
    label_encode_columns = [
        "workclass",
        "education",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native.country",
    ]
    pp = (
        Pipeline()
        .with_columns(pl.col(target_column).eq(pl.lit(">50K")))
        .label_encode(label_encode_columns)
        .concat(
            [
                Pipeline().select(target_column),
                Pipeline().gbdt.lightgbm(
                    {"objective": "binary", "metric": "binary_logloss"},
                    target_column,
                    save_dir="./lightgbm",
                ),
            ],
            how="horizontal",
        )
    )

    pp.fit_transform(train_data, valid=valid_data)
    test_pred = pp.transform(test_data)
    print(test_pred.head())

    evaluate = Pipeline().metrics.binary_classification(target_column, "prediction")
    print(evaluate.fit_transform(test_pred))


if __name__ == "__main__":
    main()
