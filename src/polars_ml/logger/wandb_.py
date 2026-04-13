from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Logger

if TYPE_CHECKING:
    import wandb


class WandbLogger(Logger):
    def __init__(self, project: str, name: str | None = None, **kwargs: Any) -> None:
        self.project = project
        self.name = name
        self.kwargs = kwargs
        self._run: wandb.Run | None = None

    def start(self) -> None:
        if self._run is not None:
            return

        import wandb

        self._run = wandb.init(project=self.project, name=self.name, **self.kwargs)

    def end(self) -> None:
        if self._run is None:
            return

        self._run.finish()
        self._run = None

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        self.run.log({key: value}, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.run.log(metrics, step=step)

    def log_param(self, key: str, value: Any) -> None:
        self.run.config.update({key: value})

    def log_params(self, params: dict[str, Any]) -> None:
        self.run.config.update(params)

    def set_tag(self, key: str, value: str) -> None:
        current_tags = self.run.tags or ()
        self.run.tags = current_tags + (f"{key}:{value}",)

    def set_tags(self, tags: dict[str, str]) -> None:
        current_tags = self.run.tags or ()
        new_tags = [f"{k}:{v}" for k, v in tags.items()]
        self.run.tags = current_tags + tuple(new_tags)

    @property
    def run(self) -> wandb.Run:
        if self._run is None:
            raise RuntimeError("No active run. Call start() first.")
        return self._run
