# voco-fishspeech

Fish Speech plugin for VOCO - multilingual text-to-speech with voice cloning.

## Install

```bash
pip install -e .
```

## Usage

```python
from voco.core import AudioRouter

router = AudioRouter()
router.load("fishspeech", alias="tts")

for result in router.infer(
    "tts",
    text="Hello world",
    ref_audio="reference.wav",
    ref_text="Reference transcript"
):
    print(result.audio.shape)
```

## Features

- Multilingual support
- Voice cloning capabilities
- High-quality speech synthesis
- Flexible reference encoding

## Requirements

- Python >=3.10
- PyTorch with CUDA support recommended
