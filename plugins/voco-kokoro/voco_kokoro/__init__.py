from voco.core.registry import register_model
from .kokoro_driver import KokoroDriver
from .model import KModel
from .pipeline import KPipeline

register_model("kokoro", KokoroDriver)

__all__ = ["KokoroDriver", "KModel", "KPipeline"]
