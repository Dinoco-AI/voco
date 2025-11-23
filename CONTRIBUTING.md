# Contributing

## Setup

```bash
git clone https://github.com/yourusername/voco.git
cd voco
pip install -e .
pip install -e plugins/voco-kokoro
pip install ruff mypy
```

## Code Style

Use ruff for formatting and linting:

```bash
ruff format .
ruff check .
```

## Adding a Plugin

Create a new plugin in `plugins/`:

```
plugins/voco-yourmodel/
├── pyproject.toml
└── voco_yourmodel/
    ├── __init__.py
    └── driver.py
```

### Plugin Structure

**driver.py:**
```python
from voco.core.base_model import BaseAudioModel

class YourModelDriver(BaseAudioModel):
    def load(self):
        # Load your model here
        self._loaded = True

    def generate(self, *args, **kwargs):
        # Run inference
        return output
```

**__init__.py:**
```python
from voco.core.registry import register_model
from .driver import YourModelDriver

register_model("yourmodel", YourModelDriver)
```

**pyproject.toml:**
```toml
[project]
name = "voco-yourmodel"
version = "0.0.1"
dependencies = [
    "voco>=0.0.1",
    # your deps here
]
```

## Testing

Test your plugin:

```python
from voco.core import AudioRouter
import voco_yourmodel

router = AudioRouter()
router.load("yourmodel", device="cpu")
output = router.infer("yourmodel", input_data)
```

## Pull Requests

1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Run `ruff format .` and `ruff check .`
5. Submit a PR
