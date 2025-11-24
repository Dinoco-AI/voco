# voco-fishspeech

Fish Speech 1.5 TTS plugin for VOCO audio inference runtime.

## Installation

```bash
cd plugins/voco-fishspeech
pip install -e .
```

## Requirements

- Fish Speech 1.5 checkpoints should be available at the configured checkpoint path
- All Fish Speech code is bundled inside the plugin (no external dependencies needed)

## Usage

```python
from voco import Voco

# Initialize with Fish Speech
voco = Voco(model="fishspeech", device="cuda", checkpoint_path="checkpoints/fish-speech-1.5")

# Generate audio
for audio_chunk in voco.generate(
    text="Hello, this is a test of Fish Speech.",
    reference_audio="path/to/reference.wav",
    reference_text="reference audio",
    temperature=0.7,
    top_p=0.7,
    repetition_penalty=1.1
):
    # Process audio chunk
    pass
```

## Configuration

- `checkpoint_path`: Path to Fish Speech 1.5 checkpoints (default: "checkpoints/fish-speech-1.5")
- `device`: Device to run inference on (default: "cuda")
- `dtype`: Data type for inference (default: "bfloat16")
- `compile`: Whether to compile the model with torch.compile (default: True)
