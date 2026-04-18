try:
    from .lightning_ import PyTorchLightning
except ModuleNotFoundError as e:
    if e.name == "lightning":
        raise e

__all__ = ["PyTorchLightning"]
