from typing import Any, Generator
from voco.core.base_model import BaseAudioModel


class NeuTTSDriver(BaseAudioModel):
    '''NeuTTS Air wrapper'''
    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float32",
        **kwargs: Any
    ) -> None:
        super().__init__(device=device, dtype=dtype, **kwargs)
        self.pipeline = None
        self._backbone_repo = kwargs.get("backbone_repo", "neuphonic/neutts-air")
        self._codec_repo = kwargs.get("codec_repo", "neuphonic/neucodec")

    def load(self) -> None:
        from .pipeline import NeuTTSPipeline

        self.pipeline = NeuTTSPipeline(
            backbone_repo=self._backbone_repo,
            codec_repo=self._codec_repo,
            device=self.device
        )
        self._loaded = True

    def generate(
        self,
        text: str,
        ref_audio: str = None,
        ref_text: str = None,
        ref_codes: Any = None,
        **kwargs: Any
    ) -> Generator[Any, None, None]:
        if not self._loaded or self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        for result in self.pipeline(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            ref_codes=ref_codes,
            **kwargs
        ):
            yield result

    def unload(self) -> None:
        if self.pipeline is not None:
            import torch
            del self.pipeline
            self.pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._loaded = False
