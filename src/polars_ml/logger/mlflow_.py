from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import Logger

if TYPE_CHECKING:
    import mlflow


class MLflowLogger(Logger):
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str | None = None,
        run_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        import mlflow

        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self.kwargs = kwargs
        self._run: mlflow.ActiveRun | None = None

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        mlflow.set_experiment(self.experiment_name)

    def start(self) -> None:
        if self._run is not None:
            return
        import mlflow

        self._run = mlflow.start_run(run_name=self.run_name, **self.kwargs)

    def end(self) -> None:
        if self._run is None:
            return

        import mlflow

        mlflow.end_run()
        self._run = None

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        import mlflow

        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        import mlflow

        mlflow.log_metrics(metrics, step=step)

    def log_param(self, key: str, value: Any) -> None:
        import mlflow

        mlflow.log_param(key, value)

    def log_params(self, params: dict[str, Any]) -> None:
        import mlflow

        mlflow.log_params(params)

    def set_tag(self, key: str, value: str) -> None:
        import mlflow

        mlflow.set_tag(key, value)

    def set_tags(self, tags: dict[str, str]) -> None:
        import mlflow

        mlflow.set_tags(tags)

    @property
    def run(self) -> mlflow.ActiveRun:
        if self._run is None:
            raise RuntimeError("No active run. Call start() first.")
        return self._run
