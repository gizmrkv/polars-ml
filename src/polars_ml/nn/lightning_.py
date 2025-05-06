from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import polars as pl
from polars import DataFrame, Series

from polars_ml.component import Component

if TYPE_CHECKING:
    import lightning as L
    import torch


class PytorchLightning(Component):
    def __init__(
        self,
        input_fn: Callable[[DataFrame], Any],
        data_module_fn: Callable[
            [DataFrame, DataFrame | Mapping[str, DataFrame] | None],
            "L.LightningDataModule",
        ],
        model: "L.LightningModule",
        trainer: "L.Trainer",
        *,
        prediction_name: str = "pytorch_lightning",
        include_input: bool = True,
    ):
        self.input_fn = input_fn
        self.datamodule_fn = data_module_fn
        self.model = model
        self.trainer = trainer
        self.prediction_name = prediction_name
        self.include_input = include_input

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> "PytorchLightning":
        datamodule = self.datamodule_fn(data, validation_data)
        self.trainer.fit(self.model, datamodule=datamodule)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        import torch

        inputs = self.input_fn(data)
        output: torch.Tensor = self.model(*inputs)
        pred = output.detach().cpu().numpy()
        if pred.ndim == 1:
            columns = [Series(self.prediction_name, pred)]
        else:
            n = pred.shape[1]
            zero_pad = len(str(n))
            columns = [
                Series(f"{self.prediction_name}_{i:0{zero_pad}d}", pred[:, i])
                for i in range(n)
            ]

        if self.include_input:
            return data.with_columns(columns)
        else:
            return DataFrame(columns)
