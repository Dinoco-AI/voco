import time
from voco.core import AudioRouter
import voco_kokoro  # Register the plugin

# Enable caching
router = AudioRouter(
    cache=True,
    cache_config={
        "max_size_mb": 100,
        "ttl_seconds": 3600,  # 1 hour
        "warn_at_percent": 80,
    }
)

# Load model
router.load("kokoro", alias="tts", device="cpu", lang_code="a")

text = "Hello world! This is a caching demonstration."

# First call - cache miss
print("First call (generating)...")
start = time.time()
audio1 = router.infer("tts", text=text, voice="af_heart")
print(f"Time: {time.time() - start:.3f}s")

# Second call - cache hit
print("\nSecond call (from cache)...")
start = time.time()
audio2 = router.infer("tts", text=text, voice="af_heart")
print(f"Time: {time.time() - start:.3f}s")

# View cache stats
print("\nCache stats:")
print(router.cache.stats())

# Test cache skip
print("\nThird call (cache=False)...")
start = time.time()
audio3 = router.infer("tts", text=text, voice="af_heart", cache=False)
print(f"Time: {time.time() - start:.3f}s")

# Clear cache
print("\nClearing cache...")
router.cache.clear()
print("Cache cleared")
print(router.cache.stats())
