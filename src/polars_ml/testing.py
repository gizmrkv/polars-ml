import pickle
import tempfile
from copy import deepcopy

import joblib
from polars import DataFrame
from polars.testing import assert_frame_equal

from polars_ml.component import Component, LazyComponent
from polars_ml.utils import Collect


def assert_fit_and_execute(
    component: Component | LazyComponent,
    *,
    fit_data: DataFrame | None = None,
    input_data: DataFrame | None = None,
    expected_data: DataFrame | None = None,
    check_fit: bool = True,
    check_execute: bool = True,
    check_execute_twice: bool = True,
    check_fit_execute: bool = True,
):
    if isinstance(component, LazyComponent):
        component = Collect(component)

    if check_fit:
        assert fit_data is not None
        component.fit(fit_data)

    if check_execute:
        assert input_data is not None and expected_data is not None
        result = component.execute(input_data)
        assert_frame_equal(result, expected_data)

    if check_execute_twice:
        assert input_data is not None and expected_data is not None
        result = component.execute(input_data)
        assert_frame_equal(result, expected_data)

    if check_fit_execute:
        assert input_data is not None and expected_data is not None
        result = component.fit_execute(input_data)
        assert_frame_equal(result, expected_data)


def assert_component(
    component: Component | LazyComponent,
    *,
    fit_data: DataFrame | None = None,
    input_data: DataFrame | None = None,
    expected_data: DataFrame | None = None,
    check_fit: bool = True,
    check_execute: bool = True,
    check_execute_twice: bool = True,
    check_fit_execute: bool = True,
    check_deepcopy: bool = True,
    check_pickle_dump_load: bool = True,
    check_joblib_dump_load: bool = True,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        for log_dir in [None, tmpdir]:
            component.set_log_dir(log_dir)

            assert_fit_and_execute(
                component,
                fit_data=fit_data,
                input_data=input_data,
                expected_data=expected_data,
                check_fit=check_fit,
                check_execute=check_execute,
                check_execute_twice=check_execute_twice,
                check_fit_execute=check_fit_execute,
            )

            if check_deepcopy:
                copied = deepcopy(component)
                assert_fit_and_execute(
                    copied,
                    fit_data=fit_data,
                    input_data=input_data,
                    expected_data=expected_data,
                    check_fit=check_fit,
                    check_execute=check_execute,
                    check_execute_twice=check_execute_twice,
                    check_fit_execute=check_fit_execute,
                )

            if check_pickle_dump_load:
                dumped = pickle.dumps(component)
                loaded = pickle.loads(dumped)
                assert_fit_and_execute(
                    loaded,
                    fit_data=fit_data,
                    input_data=input_data,
                    expected_data=expected_data,
                    check_fit=check_fit,
                    check_execute=check_execute,
                    check_execute_twice=check_execute_twice,
                    check_fit_execute=check_fit_execute,
                )

            if check_joblib_dump_load:
                with tempfile.TemporaryFile() as f:
                    joblib.dump(component, f)
                    f.seek(0)
                    loaded = joblib.load(f)
                    assert_fit_and_execute(
                        loaded,
                        fit_data=fit_data,
                        input_data=input_data,
                        expected_data=expected_data,
                        check_fit=check_fit,
                        check_execute=check_execute,
                        check_execute_twice=check_execute_twice,
                        check_fit_execute=check_fit_execute,
                    )
