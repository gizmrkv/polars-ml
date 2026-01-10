from __future__ import annotations

import inspect
from pathlib import Path
from typing import Literal

from const import END_INSERTION_MARKER, START_INSERTION_MARKER
from utils import insert_between_markers, render_call_args, render_params_sig

from polars_ml.pipeline.mixin import PipelineMixin


def render_methods() -> list[str]:
    codes = []

    # Handle write methods
    write_methods = []
    for name, obj in inspect.getmembers(PipelineMixin):
        if name.startswith("write_") and callable(obj):
            format_name = name.replace("write_", "")
            params = render_params_sig(obj)
            # Remove self from params if present (though render_params_sig usually handles it)
            # Add self back for the overload signature
            params_sig = ", ".join(
                ["self", f'format: Literal["{format_name}"]'] + params
            )

            codes.append(f"""
    @overload
    def write({params_sig}) -> Self: ...
""")
            write_methods.append(format_name)

    # Generate dispatch implementation for write
    codes.append("""
    def write(self, format: str, *args: Any, **kwargs: Any) -> Self:
        method_name = f"write_{format}"
        if not hasattr(self, method_name):
             raise ValueError(f"Unknown format: {format}")
        return getattr(self, method_name)(*args, **kwargs)
""")

    # Handle read methods
    read_methods = []
    for name, obj in inspect.getmembers(PipelineMixin):
        if name.startswith("read_") and callable(obj):
            format_name = name.replace("read_", "")
            params = render_params_sig(obj)
            # Remove self from params if present
            params_sig = ", ".join(
                ["self", f'format: Literal["{format_name}"]'] + params
            )

            codes.append(f"""
    @overload
    def read({params_sig}) -> Self: ...
""")
            read_methods.append(format_name)

    # Generate dispatch implementation for read
    codes.append("""
    def read(self, format: str, *args: Any, **kwargs: Any) -> Self:
        method_name = f"read_{format}"
        if not hasattr(self, method_name):
             raise ValueError(f"Unknown format: {format}")
        return getattr(self, method_name)(*args, **kwargs)
""")

    return codes


def insert_unified_io_methods():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    target_file = PROJECT_ROOT / Path("src/polars_ml/pipeline/mixin.py")
    codes = render_methods()
    insert_between_markers(
        target_file,
        "".join(codes),
        START_INSERTION_MARKER.format(
            prefix=" " * 4, suffix=" IN Unified IO PipelineMixin"
        ),
        END_INSERTION_MARKER.format(
            prefix=" " * 4, suffix=" IN Unified IO PipelineMixin"
        ),
    )


if __name__ == "__main__":
    insert_unified_io_methods()
