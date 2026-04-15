from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Self, Sequence

import lightning as L
import polars as pl
import polars.selectors as cs
import torch
import torch.nn as nn
from polars._typing import ColumnNameOrSelector
from torch.utils.data import DataLoader, Dataset

from polars_ml.base import Transformer
from polars_ml.exceptions import NotFittedError

Batch = tuple[dict[str, torch.Tensor], torch.Tensor] | dict[str, torch.Tensor]


class FeatureDataset(Dataset[Batch]):
    def __init__(
        self, x_dict: dict[str, torch.Tensor], y: torch.Tensor | None = None
    ) -> None:
        self.x_dict = x_dict
        self.y = y
        self.length = (
            len(next(iter(x_dict.values())))
            if x_dict
            else (len(y) if y is not None else 0)
        )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Batch:
        x = {k: v[idx] for k, v in self.x_dict.items()}
        if self.y is not None:
            return x, self.y[idx]
        return x


class TensorDataModule(L.LightningDataModule):
    def __init__(
        self,
        X_train: dict[str, torch.Tensor],
        y_train: torch.Tensor,
        X_val: dict[str, torch.Tensor],
        y_val: torch.Tensor,
        *,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self._train_dataset = FeatureDataset(X_train, y_train)
        self._val_dataset = FeatureDataset(X_val, y_val)
        self._batch_size = batch_size

    def train_dataloader(self) -> DataLoader[Batch]:
        return DataLoader(
            self._train_dataset, batch_size=self._batch_size, shuffle=True
        )

    def val_dataloader(self) -> DataLoader[Batch]:
        return DataLoader(self._val_dataset, batch_size=self._batch_size)


class SupervisedModule(L.LightningModule):
    def __init__(
        self,
        module: nn.Module,
        loss_fn: nn.Module,
        optimizer_class: type,
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()
        self.module = module
        self.loss_fn = loss_fn
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs

    def forward(self, **kwargs: torch.Tensor) -> torch.Tensor:
        return self.module(**kwargs)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        assert isinstance(batch, tuple)
        x, y = batch
        pred = self(**x)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        assert isinstance(batch, tuple)
        x, y = batch
        pred = self(**x)
        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Any:
        return self._optimizer_class(self.parameters(), **self._optimizer_kwargs)


class PyTorchLightning(Transformer):
    def __init__(
        self,
        module: nn.Module,
        target: ColumnNameOrSelector,
        prediction: str | Sequence[str],
        features: ColumnNameOrSelector | Iterable[ColumnNameOrSelector] | None = None,
        *,
        num_x_dtype: type[pl.DataType] | None = None,
        cat_x_dtype: type[pl.DataType] | None = None,
        target_dtype: type[pl.DataType] | None = None,
        loss_fn: nn.Module | None = None,
        optimizer: type = torch.optim.AdamW,
        optimizer_kwargs: dict[str, Any] | None = None,
        batch_size: int = 32,
        fit_dir: str | Path | None = None,
        **trainer_kwargs: Any,
    ) -> None:
        self._module = module
        self._target_selector = target
        self._prediction = (
            [prediction] if isinstance(prediction, str) else list(prediction)
        )
        self._features_selector = (
            features if features is not None else cs.exclude(target)
        )
        self._num_x_dtype = num_x_dtype
        self._cat_x_dtype = cat_x_dtype
        self._target_dtype = target_dtype
        self._loss_fn: nn.Module = loss_fn if loss_fn is not None else nn.MSELoss()
        self._optimizer = optimizer
        self._optimizer_kwargs: dict[str, Any] = optimizer_kwargs or {}
        self._batch_size = batch_size
        self._fit_dir = Path(fit_dir) if fit_dir else None
        self._trainer_kwargs = trainer_kwargs

        self._target: list[str] | None = None
        self._features: list[str] | None = None

    @property
    def target(self) -> list[str]:
        if self._target is None:
            raise NotFittedError()
        return self._target

    @property
    def features(self) -> list[str]:
        if self._features is None:
            raise NotFittedError()
        return self._features

    def to_batch(self, data: pl.DataFrame) -> Batch:
        features_df = data.select(*self.features)

        x_dict = {}
        num_cols = features_df.select(cs.float()).columns
        if num_cols:
            x_dict["num_x"] = features_df.select(*num_cols).to_torch(
                dtype=self._num_x_dtype
            )

        cat_cols = features_df.select(cs.integer()).columns
        if cat_cols:
            x_dict["cat_x"] = features_df.select(*cat_cols).to_torch(
                dtype=self._cat_x_dtype
            )

        has_target = self._target is not None and all(
            t in data.columns for t in self.target
        )
        if has_target:
            y = data.select(*self.target).to_torch(dtype=pl.Float32)
            return x_dict, y

        return x_dict

    def fit(self, data: pl.DataFrame, **more_data: pl.DataFrame) -> Self:
        self._target = (
            data.lazy().select(self._target_selector).collect_schema().names()
        )
        self._features = (
            data.lazy().select(self._features_selector).collect_schema().names()
        )

        # Convert train data
        train_batch = self.to_batch(data)
        assert isinstance(train_batch, tuple)
        X_train, y_train = train_batch

        lightning_module = SupervisedModule(
            self._module,
            self._loss_fn,
            self._optimizer,
            self._optimizer_kwargs,
        )
        trainer = L.Trainer(**self._trainer_kwargs)

        if more_data:
            # Use TensorDataModule when validation data is available (first entry)
            val_df = next(iter(more_data.values()))
            val_batch = self.to_batch(val_df)
            assert isinstance(val_batch, tuple)
            X_val, y_val = val_batch
            data_module = TensorDataModule(
                X_train, y_train, X_val, y_val, batch_size=self._batch_size
            )
            trainer.fit(lightning_module, data_module)
        else:
            train_dl = DataLoader(
                FeatureDataset(X_train, y_train),
                batch_size=self._batch_size,
                shuffle=True,
            )
            trainer.fit(lightning_module, train_dataloaders=train_dl)

        if self._fit_dir:
            self.save(self._fit_dir)

        return self

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        if self._features is None:
            raise NotFittedError()

        batch = self.to_batch(data)
        x_dict = batch[0] if isinstance(batch, tuple) else batch

        self._module.eval()
        with torch.no_grad():
            pred = self._module(**x_dict)

        return pl.from_numpy(pred.numpy(), schema=self._prediction)

    def save(self, fit_dir: str | Path) -> None:
        fit_dir = Path(fit_dir)
        fit_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self._module.state_dict(), fit_dir / "model.pt")
