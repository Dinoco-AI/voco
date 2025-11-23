# VOCO

Modular audio inference runtime with plugin architecture.

## Install

```bash
# Core (lightweight, no dependencies)
pip install voco

# Install plugins you need
pip install voco-kokoro  # for Kokoro TTS
```

## Usage

```python
from voco.core import AudioRouter
import voco_kokoro

router = AudioRouter()
router.load("kokoro", alias="tts", device="cpu")

for result in router.infer("tts", text="Hello world", voice="af_heart"):
    audio = result.audio
```

## Features

- Zero dependencies in core
- Plugin based architecture
- Simple API
- Type safe

## Plugins

Plugins are separate packages you install as needed:

- `voco-kokoro` - Kokoro TTS

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed setup and plugin development guide.

```bash
git clone https://github.com/yourusername/voco.git
cd voco
pip install -e .
pip install -e plugins/voco-kokoro
python examples/generate_audio.py
```

## License

MIT
