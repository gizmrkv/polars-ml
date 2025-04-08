from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping

import lightning as L
import torch
from polars import DataFrame

from .pytorch_lightning_ import PytorchLightning

if TYPE_CHECKING:
    from polars_ml import Pipeline

__all__ = ["PytorchLightning"]


class NNNameSpace:
    def __init__(self, pipeline: "Pipeline"):
        self.pipeline = pipeline

    def lightning(
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
        component_name: str | None = None,
    ) -> "Pipeline":
        return self.pipeline.pipe(
            PytorchLightning(
                input_fn,
                data_module_fn,
                model,
                trainer,
                ckpt_path=ckpt_path,
                prediction_name=prediction_name,
                include_input=include_input,
            ),
            component_name=component_name,
        )
