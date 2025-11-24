from voco.core.registry import register_model
from .fishspeech_driver import FishSpeechDriver

register_model("fishspeech", FishSpeechDriver)

__all__ = ["FishSpeechDriver"]
