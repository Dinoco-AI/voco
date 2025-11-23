from abc import ABC, abstractmethod
from typing import Any


class BaseAudioModel(ABC):
    def __init__(self, device: str = "cpu", dtype: str = "float32", **kwargs: Any) -> None:
        self.device = device
        self.dtype = dtype
        self.config = kwargs
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def generate(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self) -> None:
        self._loaded = False
