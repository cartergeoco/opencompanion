# OpenCompanion

A file-based voice companion with no frontend. Speak into the microphone and it acknowledges when it hears you talking.

## Branches

- `main` — stable releases
- `nightly` — current development

Publishing a GitHub Release (not a prerelease) fast-forwards `main` to that tag.

## Run

```powershell
pip install -r requirements.txt
Copy-Item configure.example.json configure.json
python run.py
```

List microphones:

```powershell
python run.py --devices
```

Adjust `configure.json` if it misses speech or reacts to room noise.

TTS is selected in `configure.json` under `tts.engine`: `local` or `elevenlabs`.
Local speech uses Resemble AI Chatterbox Turbo. Install it with:

```powershell
pip install -r requirements-tts.txt
```

Optional local voice cloning uses a 5+ second WAV at `tts.local.audio_prompt_path`.
ElevenLabs needs `tts.elevenlabs.api_key` or `ELEVENLABS_API_KEY`, plus a `voice_id`.
Expressive tags such as `[chuckle]`, `[laugh]`, and `[cough]` work with the local Turbo model.

## Local screen vision

Screen vision uses Moondream 2 through the local Photon inference engine. It
runs in an isolated environment so its newer PyTorch build does not interfere
with voice recognition:

```powershell
python vision.py --setup-photon
python vision.py --list-monitors
python vision.py
```

Each question captures a fresh screenshot. Press Enter for a complete UI
description, ask a specific question such as `Where is the Save button?`, or
use `python vision.py --watch 0.5` for efficient change-triggered monitoring.
Watch mode uses Windows Graphics Capture and Desktop Duplication events. It does
not take screenshots on a timer: Windows signals an actual compositor update,
the script debounces it, then captures one semantic keyframe. A low-level mouse
hook tracks pointing, circling, dragging, and likely text highlighting without
images. Continuously changing content such as video is sampled at most once per
`continuous_scene_interval_sec`. Uncertain results can be escalated to local
Qwen3-VL in the background. Frames are never written to disk. If Photon is
unavailable, the script falls back to the configured local Ollama model.

Monitor `0` captures the full virtual desktop; `1` is normally the primary
monitor. Screenshots are sent only to the configured Ollama endpoint and are
not saved to disk.
