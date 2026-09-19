from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"

DEFAULTS: dict[str, Any] = {
    "sample_rate": 16000,
    "block_ms": 30,
    "speech_threshold": 0.02,
    "start_frames": 4,
    "end_frames": 12,
    "speak_aloud": True,
    "device": None,
}


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    settings = dict(DEFAULTS)
    if path.exists():
        settings.update(json.loads(path.read_text(encoding="utf-8")))
    return settings
