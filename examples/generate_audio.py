from voco.core import AudioRouter
import voco_kokoro
import torch
from scipy.io import wavfile

# Load model
router = AudioRouter()
router.load("kokoro", alias="tts", device="cpu", lang_code="a")

# Generate audio
text = "Hello world! This is VOCO with Kokoro TTS."
audio_chunks = []
for result in router.infer("tts", text=text, voice="af_heart", speed=1.0):
    if result.audio is not None:
        audio_chunks.append(result.audio)

# Save
audio = torch.cat(audio_chunks, dim=-1)
audio_np = audio.squeeze().cpu().numpy()
wavfile.write("output.wav", 24000, audio_np)
print(f"Saved output.wav ({audio.shape[-1] / 24000:.2f}s)")
