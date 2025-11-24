from typing import Any, Generator
from voco.core.base_model import BaseAudioModel


class FishSpeechDriver(BaseAudioModel):
    """Fish Speech 1.5 wrapper for VOCO"""

    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "bfloat16",
        **kwargs: Any
    ) -> None:
        super().__init__(device=device, dtype=dtype, **kwargs)
        self.vqgan = None
        self.llama = None
        self.llama_decode = None
        self._checkpoint_path = kwargs.get("checkpoint_path", "checkpoints/fish-speech-1.5")
        self._compile = kwargs.get("compile", True)

    def load(self) -> None:
        from .pipeline import FishSpeechPipeline

        self.pipeline = FishSpeechPipeline(
            checkpoint_path=self._checkpoint_path,
            device=self.device,
            dtype=self.dtype,
            compile=self._compile
        )
        self._loaded = True

    def generate(
        self,
        text: str,
        reference_audio: str,
        reference_text: str = "reference audio",
        temperature: float = 0.7,
        top_p: float = 0.7,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 1000,
        chunk_length: int = 150,
        **kwargs: Any
    ) -> Generator[Any, None, None]:
        if not self._loaded or self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        for result in self.pipeline(
            text=text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            chunk_length=chunk_length,
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
