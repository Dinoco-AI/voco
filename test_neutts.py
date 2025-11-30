import os
import sys

if sys.platform == 'darwin':
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        lib_path = '/opt/homebrew/Cellar/espeak/1.48.04_1/lib/libespeak.1.1.48.dylib'
        if os.path.exists(lib_path):
            EspeakWrapper.set_library(lib_path)
    except Exception:
        pass

from voco.core import AudioRouter
import torch
from scipy.io import wavfile

print("Loading NeuTTS Air model...")
router = AudioRouter()
router.load(
    "neutts",
    alias="tts",
    device="cpu",
    backbone_repo="neuphonic/neutts-air",
    codec_repo="neuphonic/neucodec"
)

text = "Hello world! This is a test of the NeuTTS Air text to speech model running through VOCO."
print(f"\nGenerating audio for: '{text}'")

# Reference audio and text from plugin samples
ref_audio = "plugins/voco-neutts/samples/dave.wav"
ref_text = "plugins/voco-neutts/samples/dave.txt"

# Read reference text
with open(ref_text, 'r') as f:
    ref_text_content = f.read().strip()

print(f"Using reference voice from: {ref_audio}")
print(f"Reference text: {ref_text_content}")

audio_chunks = []
for result in router.infer(
    "tts",
    text=text,
    ref_audio=ref_audio,
    ref_text=ref_text_content
):
    if result.audio is not None:
        audio_chunks.append(result.audio)
        print(f"  Generated audio chunk: {result.audio.shape[-1]} samples")
        print(f"  Phonemes: {result.phonemes[:100]}{'...' if len(result.phonemes) > 100 else ''}")

if audio_chunks:
    audio = torch.cat(audio_chunks, dim=-1)
    print(f"\nTotal audio shape: {audio.shape}")
    print(f"Duration: {audio.shape[-1] / 24000:.2f}s")

    output_file = "output_neutts.wav"
    audio_np = audio.squeeze().cpu().numpy()
    wavfile.write(output_file, 24000, audio_np)
    print(f"Saved to: {output_file}")
else:
    print("No audio generated!")
