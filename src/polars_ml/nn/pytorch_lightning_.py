from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Self, Tuple

import lightning as L
import polars as pl
import torch
from polars import DataFrame, Series
from torch.utils.data import DataLoader, TensorDataset

from polars_ml.pipeline.component import PipelineComponent


class PytorchLightning(PipelineComponent):
    def __init__(
        self,
        input_fn: Callable[[DataFrame], tuple[torch.Tensor, ...]],
        data_module_fn: Callable[
            [DataFrame, DataFrame | Mapping[str, DataFrame] | None],
            L.LightningDataModule,
        ],
        model: L.LightningModule,
        trainer: L.Trainer,
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


class TabularDataModule(L.LightningDataModule):
    def __init__(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
        *,
        numeric_columns: Iterable[str] | None = None,
        category_columns: Iterable[str] | None = None,
        label_column: str | None = None,
        numeric_dtype: pl.DataType = pl.Float32(),
        category_dtype: pl.DataType = pl.Int64(),
        label_dtype: pl.DataType = pl.Float32(),
        batch_size: int = 1024,
    ):
        super().__init__()
        self.data = data
        self.validation_data = validation_data
        self.numeric_columns = numeric_columns
        self.category_columns = category_columns
        self.label_column = label_column
        self.numeric_dtype = numeric_dtype
        self.category_dtype = category_dtype
        self.label_dtype = label_dtype
        self.batch_size = batch_size

    def to_input(self, data: DataFrame) -> list[torch.Tensor]:
        inputs = []
        if self.numeric_columns:
            inputs.append(
                data.select(self.numeric_columns).to_torch(dtype=self.numeric_dtype)
            )
        if self.category_columns:
            inputs.append(
                data.select(self.category_columns).to_torch(dtype=self.category_dtype)
            )
        if self.label_column:
            inputs.append(
                data.select(self.label_column)
                .to_torch(dtype=self.label_dtype)
                .squeeze()
            )
        return inputs

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = TensorDataset(*self.to_input(self.data))
        if isinstance(self.validation_data, DataFrame):
            self.validation_dataset = TensorDataset(
                *self.to_input(self.validation_data)
            )
        elif isinstance(self.validation_data, Mapping):
            self.validation_dataset = [
                TensorDataset(*self.to_input(data))
                for data in self.validation_data.values()
            ]
        else:
            self.validation_dataset = None

    def train_dataloader(self) -> DataLoader[Tuple[torch.Tensor, ...]]:
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(
        self,
    ) -> (
        DataLoader[Tuple[torch.Tensor, ...]]
        | list[DataLoader[Tuple[torch.Tensor, ...]]]
        | None
    ):
        if isinstance(self.validation_dataset, TensorDataset):
            return DataLoader(
                self.validation_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
            )
        elif isinstance(self.validation_dataset, list):
            return [
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=4,
                )
                for dataset in self.validation_dataset
            ]
        else:
            raise ValueError("Required validation data.")
