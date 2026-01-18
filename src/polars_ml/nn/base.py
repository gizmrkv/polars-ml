from __future__ import annotations

from typing import Any, Callable, Mapping, Self

import polars as pl
import torch
from numpy.typing import NDArray
from polars import DataFrame

from polars_ml.base import Transformer

Batch = Mapping[str, torch.Tensor]


class Module(Transformer):
    def __init__(
        self,
        model: torch.nn.Module,
        to_batch: Callable[[DataFrame], Batch],
        loss_fn: Callable[[torch.Tensor, Batch], torch.Tensor],
        optimizer: torch.optim.Optimizer,
        epochs: int = 1,
        device: str | torch.device | None = None,
        prediction_name: str = "prediction",
    ) -> None:
        self.model = model
        self.to_batch = to_batch
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = torch.device(device) if device is not None else None
        self.prediction_name = prediction_name

    def fit(self, data: DataFrame, **more_data: DataFrame) -> Self:
        dev = self.device or next(self.model.parameters()).device
        self.model.to(dev)
        self.model.train()

        for _ in range(self.epochs):
            batch = self.to_batch(data)
            batch = {k: v.to(dev) for k, v in batch.items()}

            pred = self.model(**batch)
            loss = self.loss_fn(pred, batch)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        return self

    def transform(self, data: DataFrame) -> DataFrame:
        pred = self.predict(data)
        name = self.prediction_name

        prediction_df = pl.from_numpy(
            pred,
            schema=[name]
            if pred.ndim == 1
            else [f"{name}_{i}" for i in range(pred.shape[1])],
        )

        return pl.concat([data, prediction_df], how="horizontal")

    def _resolve_device(self) -> torch.device:
        if self.device is not None:
            return self.device
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _to_device(self, batch: Batch, dev: torch.device) -> dict[str, torch.Tensor]:
        return {k: v.to(dev) for k, v in batch.items()}

    def predict(self, data: DataFrame) -> NDArray[Any]:
        dev = self._resolve_device()
        self.model.to(dev)
        self.model.eval()

        with torch.no_grad():
            batch = self._to_device(self.to_batch(data), dev)
            pred = self.model(**batch)

        pred_np = pred.detach().to("cpu").numpy()
        if pred_np.ndim == 2 and pred_np.shape[1] == 1:
            pred_np = pred_np.reshape(-1)
        return pred_np
