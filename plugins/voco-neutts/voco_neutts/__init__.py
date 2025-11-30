from voco.core.registry import register_model
from .neutts_driver import NeuTTSDriver
from .pipeline import NeuTTSPipeline

register_model("neutts", NeuTTSDriver)

__all__ = ["NeuTTSDriver", "NeuTTSPipeline"]
