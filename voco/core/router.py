from typing import Any, Optional

from .base_model import BaseAudioModel
from .cache import VocoCache
from .registry import load as registry_load


class ModelNotLoadedError(Exception):
    pass


class AudioRouter:
    def __init__(
        self,
        cache: bool = False,
        cache_config: Optional[dict[str, Any]] = None,
    ) -> None:
        self._models: dict[str, BaseAudioModel] = {}
        self._cache: Optional[VocoCache] = None

        if cache:
            config = cache_config or {}
            self._cache = VocoCache(**config)

    @property
    def cache(self) -> Optional[VocoCache]:
        return self._cache

    def load(
        self,
        name: str,
        alias: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        **kwargs: Any,
    ) -> BaseAudioModel:
        if alias is None:
            alias = name
        if alias in self._models:
            raise ValueError(
                f"Alias '{alias}' is already in use. "
                f"Use unload('{alias}') first or choose a different alias."
            )
        model = registry_load(name=name, device=device, dtype=dtype, auto_load=True, **kwargs)
        self._models[alias] = model
        return model

    def infer(self, alias: str, *args: Any, **kwargs: Any) -> Any:
        if alias not in self._models:
            available = ", ".join(self._models.keys()) or "none"
            raise ModelNotLoadedError(
                f"Model alias '{alias}' not found. Available aliases: {available}"
            )

        use_cache = kwargs.pop("cache", True)
        text = kwargs.get("text", "")

        if self._cache and use_cache and text:
            cache_params = {k: v for k, v in kwargs.items() if k != "text"}
            cached = self._cache.get(alias, text, **cache_params)
            if cached:
                return cached

        result = self._models[alias].generate(*args, **kwargs)

        if self._cache and use_cache and text and isinstance(result, bytes):
            cache_params = {k: v for k, v in kwargs.items() if k != "text"}
            self._cache.put(alias, text, result, **cache_params)

        return result

    def get_model(self, alias: str) -> BaseAudioModel:
        if alias not in self._models:
            raise ModelNotLoadedError(f"Model alias '{alias}' not found")
        return self._models[alias]

    def unload(self, alias: str) -> None:
        if alias not in self._models:
            raise ModelNotLoadedError(f"Model alias '{alias}' not found")
        self._models[alias].unload()
        del self._models[alias]

    def unload_all(self) -> None:
        for model in self._models.values():
            model.unload()
        self._models.clear()

    def list_loaded(self) -> dict[str, BaseAudioModel]:
        return self._models.copy()

    def is_loaded(self, alias: str) -> bool:
        return alias in self._models

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, alias: str) -> bool:
        return alias in self._models
