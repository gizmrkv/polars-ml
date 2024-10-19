from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Self,
    Tuple,
    TypeVar,
    override,
)

from polars import DataFrame, LazyFrame


class BaseComponent(ABC):
    def is_fitted(self) -> bool:
        try:
            return self._is_fitted
        except AttributeError:
            self._is_fitted: bool = False
            return self._is_fitted

    @property
    def log_dir(self) -> Path | None:
        try:
            return self._log_dir
        except AttributeError:
            self._log_dir: Path | None = None
            return self._log_dir

    def set_log_dir(self, log_dir: str | Path | None) -> Self:
        if log_dir := log_dir:
            self._log_dir = Path(log_dir)
        return self

    @property
    def component_name(self) -> str:
        try:
            return self._component_name
        except AttributeError:
            self._component_name: str = self.__class__.__name__
            return self._component_name

    def set_component_name(self, name: str) -> Self:
        self._component_name = name
        return self


class Component(BaseComponent):
    def fit(self, data: DataFrame) -> Self:
        return self

    @abstractmethod
    def execute(self, data: DataFrame) -> DataFrame: ...

    def fit_execute(self, data: DataFrame) -> DataFrame:
        return self.fit(data).execute(data)


class LazyComponent(BaseComponent):
    def fit(self, data: LazyFrame) -> Self:
        return self

    @abstractmethod
    def execute(self, data: LazyFrame) -> LazyFrame: ...

    def fit_execute(self, data: LazyFrame) -> LazyFrame:
        return self.fit(data).execute(data)


ComponentType = TypeVar("ComponentType", Component, LazyComponent)


class ComponentList(Generic[ComponentType], BaseComponent):
    def __init__(self, components: List[ComponentType] | None = None):
        self.components = components or []

    def __getitem__(self, index: int) -> ComponentType:
        return self.components[index]

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self) -> Iterator[ComponentType]:
        return iter(self.components)

    def append(self, component: ComponentType):
        self.components.append(component)

    def extend(self, components: Iterable[ComponentType]):
        self.components.extend(components)

    def clear(self):
        self.components.clear()

    @override
    def is_fitted(self) -> bool:
        return all(component.is_fitted() for component in self)

    @override
    def set_log_dir(self, log_dir: str | Path | None) -> Self:
        super().set_log_dir(log_dir)
        zero_pad = len(str(len(self.components)))
        for i, component in enumerate(self.components):
            component.set_log_dir(
                self.log_dir / f"{i:0>{zero_pad}}_{component.component_name}"
                if self.log_dir
                else None
            )

        return self


class ComponentDict(Generic[ComponentType], BaseComponent):
    def __init__(self, components: Dict[str, ComponentType] | None = None):
        self.components = components or {}

    def __setitem__(self, key: str, value: ComponentType):
        self.components[key] = value

    def __getitem__(self, key: str) -> ComponentType:
        return self.components[key]

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self) -> Iterator[str]:
        return iter(self.components)

    def keys(self) -> Iterable[str]:
        return self.components.keys()

    def values(self) -> Iterable[ComponentType]:
        return self.components.values()

    def items(self) -> Iterable[Tuple[str, ComponentType]]:
        return self.components.items()

    def clear(self):
        self.components.clear()

    @override
    def is_fitted(self) -> bool:
        return all(component.is_fitted() for component in self.values())

    @override
    def set_log_dir(self, log_dir: str | Path | None) -> Self:
        super().set_log_dir(log_dir)
        for name, component in self.components.items():
            component.set_log_dir(
                self.log_dir / f"{name}_{component.component_name}"
                if self.log_dir
                else None
            )

        return self
