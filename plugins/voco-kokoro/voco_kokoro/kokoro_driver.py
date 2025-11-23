from typing import Any, Generator
from voco.core.base_model import BaseAudioModel


class KokoroDriver(BaseAudioModel):
    '''kokoro wrapper'''
    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float32",
        **kwargs: Any
    ) -> None:
        super().__init__(device=device, dtype=dtype, **kwargs)
        self.pipeline = None
        self._lang_code = kwargs.get("lang_code", "a")
        self._repo_id = kwargs.get("repo_id", "hexgrad/Kokoro-82M")

    def load(self) -> None:
        from .pipeline import KPipeline

        self.pipeline = KPipeline(
            lang_code=self._lang_code,
            repo_id=self._repo_id,
            device=self.device
        )
        self._loaded = True

    def generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        **kwargs: Any
    ) -> Generator[Any, None, None]:
        if not self._loaded or self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        for result in self.pipeline(text=text, voice=voice, speed=speed, **kwargs):
            yield result

    def unload(self) -> None:
        if self.pipeline is not None:
            import torch
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._loaded = False
