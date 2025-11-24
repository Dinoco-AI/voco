from pathlib import Path
from typing import Generator, Optional
import torch
import torchaudio
import numpy as np
import scipy.io.wavfile
from loguru import logger
from huggingface_hub import snapshot_download

# Import from internal modules
from .tools.vqgan.inference import load_model as vqgan_load_model
from .tools.llama.generate import load_model as llama_load_model, generate_long


class FishSpeechPipeline:
    """Pipeline for Fish Speech 1.5 inference"""

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/fish-speech-1.5",
        device: str = "cuda",
        dtype: str = "bfloat16",
        compile: bool = True,
        hf_repo_id: str = "fishaudio/fish-speech-1.5"
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.compile = compile
        self.hf_repo_id = hf_repo_id

        # Convert dtype string to torch dtype
        if dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif dtype == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        self.vqgan = None
        self.llama = None
        self.llama_decode = None

        # Auto-download if checkpoint doesn't exist
        self._ensure_checkpoint_exists()

        self._load_models()

    def _ensure_checkpoint_exists(self):
        """Download checkpoint from Hugging Face if it doesn't exist locally"""
        checkpoint_dir = Path(self.checkpoint_path)
        vqgan_checkpoint = checkpoint_dir / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        llama_checkpoint = checkpoint_dir / "model.pth"

        # Check if key checkpoint files exist
        if not vqgan_checkpoint.exists() or not llama_checkpoint.exists():
            logger.info(f"Checkpoint not found at {self.checkpoint_path}")
            logger.info(f"Downloading Fish Speech 1.5 from Hugging Face: {self.hf_repo_id}")

            # Create checkpoint directory
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Download from Hugging Face
            try:
                downloaded_path = snapshot_download(
                    repo_id=self.hf_repo_id,
                    local_dir=str(checkpoint_dir),
                    local_dir_use_symlinks=False
                )
                logger.info(f"✅ Downloaded checkpoint to: {downloaded_path}")
            except Exception as e:
                logger.error(f"Failed to download checkpoint: {e}")
                raise RuntimeError(
                    f"Could not download Fish Speech 1.5 checkpoint from {self.hf_repo_id}. "
                    f"Please check your internet connection or download manually."
                ) from e
        else:
            logger.info(f"Using existing checkpoint at: {self.checkpoint_path}")

    def _load_models(self):
        """Load VQGAN and LLAMA models"""
        logger.info("Loading Fish Speech VQGAN model...")
        vqgan_checkpoint = f"{self.checkpoint_path}/firefly-gan-vq-fsq-8x1024-21hz-generator.pth"
        self.vqgan = vqgan_load_model(
            config_name="firefly_gan_vq",
            checkpoint_path=vqgan_checkpoint,
            device=self.device,
        )
        logger.info("VQGAN model loaded")

        logger.info("Loading Fish Speech LLAMA model...")
        self.llama, self.llama_decode = llama_load_model(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            precision=self.dtype,
            compile=self.compile,
        )

        with torch.device(self.device):
            self.llama.setup_caches(
                max_batch_size=1,
                max_seq_len=self.llama.config.max_seq_len,
                dtype=next(self.llama.parameters()).dtype,
            )
        logger.info("LLAMA model loaded")

    def __call__(
        self,
        text: str,
        reference_audio: str,
        reference_text: str = "reference audio",
        temperature: float = 0.7,
        top_p: float = 0.7,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 1000,
        chunk_length: int = 150,
        num_samples: int = 1,
        iterative_prompt: bool = False,
        **kwargs
    ) -> Generator[np.ndarray, None, None]:
        """
        Generate audio from text using Fish Speech 1.5

        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio file
            reference_text: Text description of reference audio
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            max_new_tokens: Maximum tokens to generate
            chunk_length: Text chunk length for generation
            num_samples: Number of samples to generate
            iterative_prompt: Whether to use iterative prompting

        Yields:
            Generated audio as numpy arrays
        """
        # Load and process reference audio
        if isinstance(reference_audio, str):
            reference_audio = [reference_audio]

        audio, sr = torchaudio.load(reference_audio[0])
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)

        desired_sr = self.vqgan.spec_transform.sample_rate
        if sr != desired_sr:
            audio = torchaudio.functional.resample(audio, sr, desired_sr)

        audio = audio.to(self.device)
        audio_lengths = torch.tensor([audio.shape[-1]], device=self.device, dtype=torch.long)

        # Encode reference audio to tokens
        indices_tuple = self.vqgan.encode(audio[None], audio_lengths)
        reference_tokens = indices_tuple[0][0]

        # Generate speech codes
        with torch.no_grad():
            generator = generate_long(
                model=self.llama,
                device=self.device,
                decode_one_token=self.llama_decode,
                text=text.strip(),
                prompt_text=[reference_text],
                prompt_tokens=[reference_tokens],
                num_samples=num_samples,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                compile=self.compile,
                iterative_prompt=iterative_prompt,
                chunk_length=chunk_length,
            )

            codes = []
            for response in generator:
                if response.action == "sample":
                    codes.append(response.codes.detach().cpu().numpy())
                elif response.action == "next":
                    if codes:
                        # Concatenate all codes
                        all_codes = np.concatenate(codes, axis=1)
                        all_codes_torch = torch.from_numpy(all_codes).long().to(self.device).unsqueeze(0)
                        feature_lengths = torch.tensor([all_codes_torch.shape[-1]], device=self.device)

                        # Decode to audio
                        with torch.no_grad():
                            fake_audio, _ = self.vqgan.decode(
                                indices=all_codes_torch,
                                feature_lengths=feature_lengths
                            )

                        audio_np = fake_audio[0, 0].float().detach().cpu().numpy()
                        yield audio_np
                        codes = []

            # Handle any remaining codes
            if codes:
                all_codes = np.concatenate(codes, axis=1)
                all_codes_torch = torch.from_numpy(all_codes).long().to(self.device).unsqueeze(0)
                feature_lengths = torch.tensor([all_codes_torch.shape[-1]], device=self.device)

                with torch.no_grad():
                    fake_audio, _ = self.vqgan.decode(
                        indices=all_codes_torch,
                        feature_lengths=feature_lengths
                    )

                audio_np = fake_audio[0, 0].float().detach().cpu().numpy()
                yield audio_np
