# voco-kokoro

Kokoro TTS plugin for VOCO - fast, lightweight text-to-speech.

## Install

```bash
pip install -e .
```

## Usage

```python
from voco.core import AudioRouter

router = AudioRouter()
router.load("kokoro", alias="tts", lang_code="a")

for result in router.infer("tts", text="Hello world", voice="af_heart", speed=1.0):
    print(result.audio.shape)
```

## Features

- Fast inference on CPU
- Multiple voices and languages
- Streaming support
- 82M parameter model

## Requirements

- Python >=3.10
- espeak-ng installed on system
