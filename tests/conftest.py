import sys
import types

# Mock livekit.plugins.noise_cancellation
noise_cancellation = types.ModuleType("noise_cancellation")
silero = types.ModuleType("silero")

plugins = types.ModuleType("livekit.plugins")
plugins.noise_cancellation = noise_cancellation
plugins.silero = silero

livekit = types.ModuleType("livekit")
livekit.plugins = plugins

sys.modules["livekit"] = livekit
sys.modules["livekit.plugins"] = plugins
sys.modules["livekit.plugins.noise_cancellation"] = noise_cancellation
sys.modules["livekit.plugins.silero"] = silero
