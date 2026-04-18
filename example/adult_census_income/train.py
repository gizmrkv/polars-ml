from pathlib import Path

import kagglehub
import polars as pl

import polars_ml as pml
from polars_ml.model_selection import train_test_split


def main():
    path = Path(kagglehub.dataset_download("uciml/adult-census-income"))
    target_column = "income"
    data = pl.read_csv(path / "*.csv").with_columns(
        pl.col(target_column).eq(">50K").cast(pl.UInt8)
    )

    train_valid_idx, test_idx = train_test_split(data, 0.2, shuffle=True, seed=42)
    train_valid_data = data.select(pl.all().gather(train_valid_idx))
    test_data = data.select(pl.all().gather(test_idx))

    train_idx, valid_idx = train_test_split(
        train_valid_data, 0.4, shuffle=True, seed=42
    )
    valid_data = train_valid_data.select(pl.all().gather(valid_idx))
    train_data = train_valid_data.select(pl.all().gather(train_idx))

    pp = pml.Pipeline().label_encode(
        "workclass",
        "education",
        "marital.status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native.country",
    )

    fit_dir = Path("./outputs")
    pp.concat(
        [
            pml.Pipeline().select(target_column),
            pml.Pipeline().pipe(
                pml.gbdt.LightGBM(
                    target_column,
                    params={
                        "objective": "binary",
                        "metric": "binary_logloss",
                        "n_iterations": 500,
                    },
                    fit_dir=fit_dir / "lightgbm",
                )
            ),
        ],
        how="horizontal",
    )

    pp.fit_transform(train_data, valid=valid_data)
    test_pred = pp.transform(test_data)
    print(test_pred.head())

    evaluate = pml.metrics.BinaryClassificationMetrics(target_column, "prediction")
    print(evaluate.fit_transform(test_pred))


if __name__ == "__main__":
    main()
