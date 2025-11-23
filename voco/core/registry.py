from typing import Any

from .base_model import BaseAudioModel
from .device import get_device, get_dtype

_MODEL_REGISTRY: dict[str, type[BaseAudioModel]] = {}


class ModelAlreadyRegisteredError(Exception):
    pass


class ModelNotFoundError(Exception):
    pass


def register_model(name: str, model_cls: type[BaseAudioModel]) -> None:
    if name in _MODEL_REGISTRY:
        raise ModelAlreadyRegisteredError(
            f"Model '{name}' is already registered. "
            f"Existing: {_MODEL_REGISTRY[name]}, New: {model_cls}"
        )
    if not issubclass(model_cls, BaseAudioModel):
        raise TypeError(f"Model class must inherit from BaseAudioModel. Got: {model_cls}")
    _MODEL_REGISTRY[name] = model_cls


def unregister_model(name: str) -> None:
    if name not in _MODEL_REGISTRY:
        raise ModelNotFoundError(f"Model '{name}' not found in registry")
    del _MODEL_REGISTRY[name]


def get_registered_models() -> dict[str, type[BaseAudioModel]]:
    return _MODEL_REGISTRY.copy()


def is_registered(name: str) -> bool:
    return name in _MODEL_REGISTRY


def load(
    name: str,
    device: str | None = None,
    dtype: str | None = None,
    auto_load: bool = True,
    **kwargs: Any,
) -> BaseAudioModel:
    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys()) or "none"
        raise ModelNotFoundError(
            f"Model '{name}' not found in registry. Available models: {available}"
        )
    model_cls = _MODEL_REGISTRY[name]
    device = get_device(device)
    dtype = get_dtype(dtype)
    model = model_cls(device=device, dtype=dtype, **kwargs)
    if auto_load:
        model.load()
    return model


def clear_registry() -> None:
    _MODEL_REGISTRY.clear()
