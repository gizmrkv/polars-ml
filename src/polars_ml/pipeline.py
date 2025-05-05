from typing import Any, Callable, Iterable, Iterator, Mapping

from polars import DataFrame

from .component import Component
from .utility import Apply


class Pipeline(Component):
    def __init__(self, *components: Component | Callable[[DataFrame], DataFrame | Any]):
        self.components: list[Component] = []
        self.extend(components)

    def fit(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> "Pipeline":
        if len(self.components) == 0:
            return self

        for component in self.components[:-1]:
            data = component.fit_transform(data, validation_data)
            if validation_data is None:
                continue

            if isinstance(validation_data, DataFrame):
                validation_data = component.transform(validation_data)
            else:
                validation_data = {
                    name: component.transform(valid)
                    for name, valid in validation_data.items()
                }

        self.components[-1].fit(data, validation_data)
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        for component in self.components:
            data = component.transform(data)

        return data

    def fit_transform(
        self,
        data: DataFrame,
        validation_data: DataFrame | Mapping[str, DataFrame] | None = None,
    ) -> DataFrame:
        if len(self.components) == 0:
            return data

        for component in self.components[:-1]:
            data = component.fit_transform(data, validation_data)
            if validation_data is None:
                continue

            if isinstance(validation_data, DataFrame):
                validation_data = component.transform(validation_data)
            else:
                validation_data = {
                    name: component.transform(valid)
                    for name, valid in validation_data.items()
                }

        return self.components[-1].fit_transform(data, validation_data)

    def append(
        self, component: Component | Callable[[DataFrame], DataFrame | Any]
    ) -> "Pipeline":
        if isinstance(component, Callable):
            component = Apply(component)
        self.components.append(component)
        return self

    def extend(
        self, components: Iterable[Component | Callable[[DataFrame], DataFrame | Any]]
    ) -> "Pipeline":
        for component in components:
            self.append(component)
        return self

    def __getitem__(self, index: int) -> Component:
        return self.components[index]

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self) -> Iterator[Component]:
        return iter(self.components)
