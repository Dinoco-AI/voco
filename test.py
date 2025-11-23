from voco.core import AudioRouter
import voco_kokoro
import torch
import numpy as np
from scipy.io import wavfile

print("Loading Kokoro TTS model...")
router = AudioRouter()
router.load("kokoro", alias="tts", device="cpu", lang_code="a", repo_id="hexgrad/Kokoro-82M")

text = "Hello world! This is a test of the Kokoro text to speech model running through VOCO."
print(f"\nGenerating audio for: '{text}'")

audio_chunks = []
for result in router.infer("tts", text=text, voice="af_heart", speed=1.0):
    if result.audio is not None:
        audio_chunks.append(result.audio)
        print(f"  Chunk: '{result.graphemes}' -> {result.audio.shape[-1]} samples")

if audio_chunks:
    audio = torch.cat(audio_chunks, dim=-1)
    print(f"\nTotal audio shape: {audio.shape}")
    print(f"Duration: {audio.shape[-1] / 24000:.2f}s")

    output_file = "output.wav"
    audio_np = audio.squeeze().cpu().numpy()
    wavfile.write(output_file, 24000, audio_np)
    print(f"Saved to: {output_file}")
else:
    print("No audio generated!")
