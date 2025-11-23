# Quickstart

## Installation

```bash
pip install voco voco-kokoro
```

## Basic Usage

```python
from voco.core import AudioRouter
import voco_kokoro
import torch
from scipy.io import wavfile

# Load model
router = AudioRouter()
router.load("kokoro", alias="tts", device="cpu", lang_code="a")

# Generate
text = "Hello world"
audio_chunks = []
for result in router.infer("tts", text=text, voice="af_heart", speed=1.0):
    if result.audio is not None:
        audio_chunks.append(result.audio)

# Save
audio = torch.cat(audio_chunks, dim=-1)
audio_np = audio.squeeze().cpu().numpy()
wavfile.write("output.wav", 24000, audio_np)
```

## Available Options

**Voices:**
- `af_heart` (American Female)
- `af_bella` (American Female)
- `am_adam` (American Male)
- `bf_emma` (British Female)
- `bm_george` (British Male)

**Languages:**
- `a` (American English)
- `b` (British English)
- `e` (Spanish)
- `f` (French)
- `j` (Japanese)
- `z` (Chinese)

**Speed:**
- `0.5` (slow)
- `1.0` (normal)
- `2.0` (fast)

## Next Steps

See [examples/](../examples/) for more usage patterns.
