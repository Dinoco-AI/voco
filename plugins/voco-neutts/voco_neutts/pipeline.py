from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Union
from loguru import logger
import torch
import numpy as np


@dataclass
class NeuTTSResult:
    """Result from NeuTTS pipeline inference"""
    graphemes: str
    phonemes: str = ""
    audio: Optional[torch.FloatTensor] = None

    def __iter__(self):
        yield self.graphemes
        yield self.phonemes
        yield self.audio


class NeuTTSPipeline:
    """
    NeuTTS Air pipeline wrapper for VOCO.

    Provides voice cloning capabilities using reference audio.
    """

    def __init__(
        self,
        backbone_repo: str = "neuphonic/neutts-air",
        codec_repo: str = "neuphonic/neucodec",
        device: Optional[str] = None
    ):
        """Initialize NeuTTS Air pipeline.

        Args:
            backbone_repo: HuggingFace repo for the backbone model
            codec_repo: HuggingFace repo for the codec
            device: Device to run on ('cpu', 'cuda', 'mps')
        """
        from .neutts import NeuTTSAir

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        # Map VOCO device names to NeuTTS device names
        backbone_device = 'gpu' if device == 'cuda' else device
        codec_device = device if device != 'mps' else 'cpu'  # NeuTTS may not support MPS for codec

        logger.info(f"Initializing NeuTTS Air with backbone on {backbone_device}, codec on {codec_device}")

        self.tts = NeuTTSAir(
            backbone_repo=backbone_repo,
            backbone_device=backbone_device,
            codec_repo=codec_repo,
            codec_device=codec_device
        )
        self.device = device
        self._ref_codes_cache = {}

    def encode_reference(self, ref_audio_path: Union[str, Path]) -> torch.Tensor:
        """Encode reference audio for voice cloning.

        Args:
            ref_audio_path: Path to reference audio file

        Returns:
            Encoded reference codes
        """
        ref_audio_path = str(ref_audio_path)

        # Check cache
        if ref_audio_path in self._ref_codes_cache:
            logger.debug(f"Using cached reference codes for {ref_audio_path}")
            return self._ref_codes_cache[ref_audio_path]

        logger.debug(f"Encoding reference audio: {ref_audio_path}")
        ref_codes = self.tts.encode_reference(ref_audio_path)

        # Cache the codes
        self._ref_codes_cache[ref_audio_path] = ref_codes
        return ref_codes

    def __call__(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        ref_codes: Optional[Union[torch.Tensor, np.ndarray]] = None,
        **kwargs
    ) -> Generator[NeuTTSResult, None, None]:
        """Generate speech from text using reference voice.

        Args:
            text: Input text to synthesize
            ref_audio: Path to reference audio file (for voice cloning)
            ref_text: Transcript of reference audio
            ref_codes: Pre-encoded reference codes (if already encoded)
            **kwargs: Additional arguments

        Yields:
            NeuTTSResult containing generated audio
        """
        # Handle reference encoding
        if ref_codes is None:
            if ref_audio is None:
                raise ValueError("Either ref_audio or ref_codes must be provided")
            ref_codes = self.encode_reference(ref_audio)

        # Load reference text if it's a file path
        if ref_text and Path(ref_text).exists():
            with open(ref_text, 'r') as f:
                ref_text = f.read().strip()

        if not ref_text:
            raise ValueError("ref_text must be provided")

        logger.debug(f"Generating speech for: {text[:50]}{'...' if len(text) > 50 else ''}")

        # Generate audio
        wav = self.tts.infer(text, ref_codes, ref_text)

        # Convert to torch tensor if needed
        if isinstance(wav, np.ndarray):
            audio = torch.from_numpy(wav).unsqueeze(0)
        else:
            audio = wav

        # Get phonemes (NeuTTS uses phonemizer internally)
        phonemes = self.tts._to_phones(text)

        yield NeuTTSResult(
            graphemes=text,
            phonemes=phonemes,
            audio=audio
        )
