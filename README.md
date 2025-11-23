<div align="left" style="margin: 10px 0 20px 0;">
  <img
    width="250"
    alt="VOCO Logo"
    src="https://github.com/user-attachments/assets/9c69bc29-22b9-4914-9fc3-b5e650eb703d"
    style="
      border-radius: 16px;
      padding: 6px;
      background: #000;
      box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    "
  />
</div>

# VOCO
<p style="
  font-size: 1.2rem;
  color: #444;
  margin: 6px 0 18px 0;
  line-height: 1.55;
  max-width: 540px;
">
  Modular audio inference runtime with plugin architecture.
</p>


## Overview

Voco separates the core runtime from model implementations. Install only what you need, use a consistent API across different models.

**Core concept:** One interface for multiple TTS/audio models.

```python
router = AudioRouter()
router.load("kokoro", alias="tts")
router.infer("tts", text="Hello world")
```

Switch models without changing your code:

```python
router.load("other-model", alias="tts")  # Same interface
router.infer("tts", text="Hello world")
```

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

## How It Works

Voco separates the **core runtime** from **model plugins**:

1. **Core** (`voco`): Router, caching, plugin loader - no heavy dependencies
2. **Plugins** (`voco-kokoro`, `voco-gtts`, etc.): Each model is a separate package with its own dependencies

Load models dynamically at runtime:

```python
router = AudioRouter()
router.load("kokoro", alias="tts")  # Loads voco-kokoro plugin
audio = router.infer("tts", text="Hello world")
```

Switch models without changing code:

```python
router.load("gtts", alias="tts")  # Replace with Google TTS
audio = router.infer("tts", text="Hello world")  # Same interface
```

## Features

- Zero dependencies in core
- Consistent API across models
- Plugin architecture
- Optional caching layer (experimental)
- Type safe

## Caching (Experimental)

Optional file-based cache for repeated inference calls.

```python
router = AudioRouter(cache=True)

# First call generates and caches
audio = router.infer("tts", text="Hello world")

# Subsequent calls return cached result
audio = router.infer("tts", text="Hello world")
```

### Configuration

```python
router = AudioRouter(
    cache=True,
    cache_config={
        "max_size_mb": 500,           # Max cache size
        "ttl_seconds": 86400,         # Time to live (1 hour - 30 days)
        "warn_at_percent": 80,        # Warning threshold
    }
)
```

### Management

```python
router.cache.stats()              # View cache usage
router.cache.clear()              # Clear all cache
router.cache.clear(model="tts")   # Clear specific model
```

### Per-Call Control

```python
# Skip cache for specific call
audio = router.infer("tts", text="Hello", cache=False)
```

### Notes

voco uses file-based cache and stored in `~/.voco/cache/`. Keys are generated from model name, text, and parameters by default.

Useful for repeated phrases. Not recommended for unique text or privacy-sensitive content and realtime environments.

## Plugins

Each plugin is a separate PyPI package with its own dependencies. Install only what you need.

### Available Plugins

- **voco-kokoro** - Kokoro TTS

### Creating Plugins

Plugins register themselves via Python entry points. See [CONTRIBUTING.md](CONTRIBUTING.md) for the plugin development guide.

```python
# Your plugin structure
voco-myplugin/
├── voco_myplugin/
│   └── __init__.py  # Implements BaseAudioModel
└── pyproject.toml   # Defines entry point
```

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
