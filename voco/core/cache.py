import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional


class VocoCache:
    def __init__(
        self,
        cache_dir: str = "~/.voco/cache",
        max_size_mb: int = 500,
        ttl_seconds: int = 2592000,
        warn_at_percent: int = 80,
    ):
        if ttl_seconds < 3600 or ttl_seconds > 2592000:
            raise ValueError("TTL must be between 1 hour (3600s) and 30 days (2592000s)")

        self.cache_dir = Path(cache_dir).expanduser()
        self.max_size = max_size_mb * 1024 * 1024
        self.warn_threshold = int(self.max_size * (warn_at_percent / 100))
        self.ttl = ttl_seconds

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_key(self, model: str, text: str, params: dict[str, Any]) -> str:
        cleaned = {k: v for k, v in params.items() if v is not None}
        param_str = json.dumps(cleaned, sort_keys=True)
        key_str = f"{model}::{text}::{param_str}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _get_cache_path(self, model: str, key: str) -> Path:
        return self.cache_dir / model / f"{key}.wav"

    def _get_total_size(self) -> int:
        if not self.cache_dir.exists():
            return 0
        total = 0
        for model_dir in self.cache_dir.iterdir():
            if model_dir.is_dir():
                for file in model_dir.glob("*.wav"):
                    total += file.stat().st_size
        return total

    def _check_and_warn(self) -> None:
        total_size = self._get_total_size()
        if total_size >= self.max_size:
            print(
                f"WARNING: Cache full ({total_size / 1024 / 1024:.1f}MB). "
                f"Auto-purging oldest entries..."
            )
            self._purge_oldest(target_size=int(self.max_size * 0.7))
        elif total_size >= self.warn_threshold:
            print(
                f"WARNING: Cache at {total_size / 1024 / 1024:.1f}MB / {self.max_size / 1024 / 1024:.0f}MB. "
                f"Run cache.clear() to free space."
            )

    def _purge_oldest(self, target_size: int) -> None:
        if not self.cache_dir.exists():
            return

        files = []
        for model_dir in self.cache_dir.iterdir():
            if model_dir.is_dir():
                for file in model_dir.glob("*.wav"):
                    files.append((file, file.stat().st_mtime))

        files.sort(key=lambda x: x[1])

        current_size = self._get_total_size()
        for file, _ in files:
            if current_size <= target_size:
                break
            current_size -= file.stat().st_size
            file.unlink()

    def get(self, model: str, text: str, **params: Any) -> Optional[bytes]:
        key = self._make_key(model, text, params)
        cache_path = self._get_cache_path(model, key)

        if not cache_path.exists():
            return None

        age = time.time() - cache_path.stat().st_mtime
        if age > self.ttl:
            cache_path.unlink()
            return None

        return cache_path.read_bytes()

    def put(self, model: str, text: str, audio: bytes, **params: Any) -> None:
        self._check_and_warn()

        key = self._make_key(model, text, params)
        cache_path = self._get_cache_path(model, key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(audio)

    def clear(self, model: Optional[str] = None) -> None:
        if model:
            model_dir = self.cache_dir / model
            if model_dir.exists():
                for file in model_dir.glob("*.wav"):
                    file.unlink()
                model_dir.rmdir()
        else:
            for model_dir in self.cache_dir.iterdir():
                if model_dir.is_dir():
                    for file in model_dir.glob("*.wav"):
                        file.unlink()
                    try:
                        model_dir.rmdir()
                    except OSError:
                        pass

    def stats(self) -> dict[str, Any]:
        total_size = 0
        total_entries = 0
        models: dict[str, dict[str, Any]] = {}

        if not self.cache_dir.exists():
            return {
                "total_size_mb": 0.0,
                "max_size_mb": round(self.max_size / 1024 / 1024, 2),
                "usage_percent": 0.0,
                "total_entries": 0,
                "ttl_hours": round(self.ttl / 3600, 1),
                "models": {},
            }

        for model_dir in self.cache_dir.iterdir():
            if model_dir.is_dir():
                model = model_dir.name
                model_size = 0
                model_entries = 0

                for file in model_dir.glob("*.wav"):
                    size = file.stat().st_size
                    total_size += size
                    model_size += size
                    total_entries += 1
                    model_entries += 1

                if model_entries > 0:
                    models[model] = {
                        "size_mb": round(model_size / 1024 / 1024, 2),
                        "entries": model_entries,
                    }

        return {
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "max_size_mb": round(self.max_size / 1024 / 1024, 2),
            "usage_percent": round(total_size * 100 / self.max_size, 1) if self.max_size > 0 else 0,
            "total_entries": total_entries,
            "ttl_hours": round(self.ttl / 3600, 1),
            "models": models,
        }
