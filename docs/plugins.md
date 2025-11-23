# Plugin System

VOCO uses a plugin architecture where the core is minimal and models are separate packages.

## Core Components

### BaseAudioModel

All plugins inherit from `BaseAudioModel`:

```python
from voco.core.base_model import BaseAudioModel

class MyModel(BaseAudioModel):
    def load(self):
        # Load model weights
        self._loaded = True

    def generate(self, *args, **kwargs):
        # Run inference
        return output
```

### Registry

Register your model:

```python
from voco.core.registry import register_model
from .driver import MyModel

register_model("mymodel", MyModel)
```

### Router

Use multiple models:

```python
from voco.core import AudioRouter

router = AudioRouter()
router.load("model1", alias="tts")
router.load("model2", alias="vc")

tts_output = router.infer("tts", text="hello")
vc_output = router.infer("vc", audio=tts_output)
```

## Plugin Structure

```
plugins/voco-mymodel/
├── pyproject.toml
└── voco_mymodel/
    ├── __init__.py
    ├── driver.py
    └── ... (model files)
```

## Publishing

Plugins are independent packages:

```bash
cd plugins/voco-mymodel
python -m build
python -m twine upload dist/*
```

Users install separately:

```bash
pip install voco
pip install voco-mymodel
```

## Examples

See [voco-kokoro](../plugins/voco-kokoro/) for a complete plugin implementation.
