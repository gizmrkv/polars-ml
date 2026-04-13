from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, Self


class Logger(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def end(self) -> None: ...

    @abstractmethod
    def log_metric(self, key: str, value: float, step: int | None = None) -> None: ...

    @abstractmethod
    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None: ...

    @abstractmethod
    def log_param(self, key: str, value: Any) -> None: ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None: ...

    @abstractmethod
    def set_tag(self, key: str, value: str) -> None: ...

    @abstractmethod
    def set_tags(self, tags: dict[str, str]) -> None: ...

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: BaseException | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ):
        self.end()
