import numpy as np
import polars as pl


def get_mock_data():
    np.random.seed(42)
    data = pl.DataFrame(
        {
            "feature1": np.random.rand(100),
            "feature2": np.random.rand(100),
            "target": np.random.randint(0, 2, 100),
        }
    )
    return data
