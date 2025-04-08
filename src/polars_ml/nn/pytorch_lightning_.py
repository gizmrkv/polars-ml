from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Mapping, Optional, Self, Tuple

import polars as pl
from polars import DataFrame, Series

from polars_ml.pipeline.component import PipelineComponent

if TYPE_CHECKING:
    import lightning as L
    import torch


class PytorchLightning(PipelineComponent):
    def __init__(
        self,
        input_fn: Callable[[DataFrame], tuple["torch.Tensor", ...]],
        data_module_fn: Callable[
            [DataFrame, DataFrame | Mapping[str, DataFrame] | None],
            "L.LightningDataModule",
        ],
        model: "L.LightningModule",
        trainer: "L.Trainer",
        *,
        ckpt_path: str | Path | None = None,
        prediction_name: str = "pytorch_lightning",
        include_input: bool = True,
    ):
        self.input_fn = input_fn
        self.datamodule_fn = data_module_fn
        self.model = model
        self.trainer = trainer
        self.ckpt_path = ckpt_path
        self.prediction_name = prediction_name
        self.include_input = include_input

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> Self:
        datamodule = self.datamodule_fn(data, validation_data)
        self.trainer.fit(self.model, datamodule=datamodule, ckpt_path=self.ckpt_path)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        import torch

        inputs = self.input_fn(data)
        output: torch.Tensor = self.model(*inputs)
        if output.ndim == 1:
            columns = [Series(self.prediction_name, output.detach().cpu().numpy())]
        else:
            columns = [
                Series(
                    f"{self.prediction_name}_{i}", output[:, i].detach().cpu().numpy()
                )
                for i in range(output.shape[1])
            ]

        if self.include_input:
            return data.with_columns(columns)
        else:
            return DataFrame(columns)
