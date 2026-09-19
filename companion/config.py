from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from companion.speech_tags import SPEECH_TAG_DEFAULTS

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configure.json"
LEGACY_CONFIG_PATH = ROOT / "config.json"

TTS_DEFAULTS: dict[str, Any] = {
    "engine": "local",
    "local": {
        "model": "chatterbox-turbo",
        "device": "auto",
        "audio_prompt_path": "",
        "temperature": 0.8,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
    },
    "elevenlabs": {
        "api_key": "",
        "voice_id": "JBFqnCBsd6RMkjVDRZzb",
        "model": "eleven_v3",
        "base_url": "https://api.elevenlabs.io",
        "output_format": "pcm_24000",
        "stability": 0.45,
        "similarity_boost": 0.75,
        "style": 0.4,
        "speed": 1.0,
    },
}

DEFAULTS: dict[str, Any] = {
    "sample_rate": 16000,
    "block_ms": 30,
    "speech_threshold": 0.02,
    "start_frames": 4,
    "end_frames": 12,
    "speak_aloud": True,
    "device": None,
    "tts": copy.deepcopy(TTS_DEFAULTS),
    "text_generation": {
        "speech_tags": copy.deepcopy(SPEECH_TAG_DEFAULTS),
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _flatten_audio(settings: dict[str, Any]) -> dict[str, Any]:
    audio = settings.get("audio")
    if isinstance(audio, dict):
        for key, value in audio.items():
            settings.setdefault(key, value)
    return settings


def load_config(path: Path | None = None) -> dict[str, Any]:
    settings = copy.deepcopy(DEFAULTS)
    config_path = path
    if config_path is None:
        config_path = CONFIG_PATH if CONFIG_PATH.exists() else LEGACY_CONFIG_PATH
    if config_path.exists():
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            settings = deep_merge(settings, loaded)
    return _flatten_audio(settings)
