from typing import Any


class ModelConfig:
    def __init__(self, **kwargs: Any) -> None:
        self._config: dict[str, Any] = kwargs

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._config[key] = value

    def update(self, **kwargs: Any) -> None:
        self._config.update(kwargs)

    def to_dict(self) -> dict[str, Any]:
        return self._config.copy()

    def __repr__(self) -> str:
        return f"ModelConfig({self._config})"

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._config[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._config


def merge_configs(*configs: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for config in configs:
        if config is not None:
            merged.update(config)
    return merged
