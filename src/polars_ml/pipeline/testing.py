from polars import DataFrame

from .component import PipelineComponent


def assert_component_valid(
    component: PipelineComponent,
    data: DataFrame,
    expected: DataFrame | None = None,
    validation_data: DataFrame | None = None,
    validation_expected: DataFrame | None = None,
    *,
    n_trials: int = 1,
    is_deterministic: bool = True,
    is_equivalent_fit_transform: bool = False,
    is_inplace: bool = False,
):
    from polars.testing import assert_frame_equal

    self = component.fit(
        data.clone(), validation_data.clone() if validation_data is not None else None
    )
    assert self is component

    data_copy = data.clone()
    out = component.transform(data_copy)

    if expected is not None:
        assert_frame_equal(out, expected)

    if validation_data is not None:
        valid_out = component.transform(validation_data.clone())
        if validation_expected is not None:
            assert_frame_equal(valid_out, validation_expected)

    if not is_inplace:
        assert_frame_equal(data_copy, data)

    if is_deterministic:
        for _ in range(n_trials):
            assert_frame_equal(out, component.transform(data.clone()))

    if is_equivalent_fit_transform:
        assert_frame_equal(out, component.fit_transform(data.clone()))
