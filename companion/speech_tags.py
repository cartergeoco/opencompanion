from __future__ import annotations

from typing import Any

CHATTERBOX_EMOTION_TAGS = (
    "happy",
    "angry",
    "surprised",
    "fear",
    "crying",
    "sarcastic",
    "dramatic",
    "whispering",
    "narration",
)
CHATTERBOX_SOUND_TAGS = (
    "laugh",
    "chuckle",
    "cough",
    "sigh",
    "gasp",
    "groan",
    "sniff",
    "clear throat",
    "shush",
)
ELEVENLABS_EMOTION_TAGS = (
    "happy",
    "sad",
    "excited",
    "angry",
    "surprised",
    "curious",
    "sarcastic",
    "annoyed",
    "thoughtful",
    "whispering",
    "mischievously",
)
ELEVENLABS_SOUND_TAGS = (
    "laughs",
    "chuckles",
    "sighs",
    "clears throat",
    "exhales",
    "swallows",
    "cough",
)

SPEECH_TAG_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "vocabulary": "auto",
    "max_tags_per_reply": 3,
    "emotion": list(CHATTERBOX_EMOTION_TAGS),
    "sound": list(CHATTERBOX_SOUND_TAGS),
}


def _as_tags(values: Any, fallback: tuple[str, ...]) -> list[str]:
    if not isinstance(values, list) or not values:
        source = fallback
    else:
        source = values
    tags: list[str] = []
    seen: set[str] = set()
    for raw in source:
        tag = " ".join(str(raw).strip().lower().replace("_", " ").split())
        tag = tag.strip("[]")
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def resolve_vocabulary(
    speech_tags: dict[str, Any] | None,
    tts_settings: dict[str, Any] | None = None,
) -> str:
    settings = speech_tags or {}
    requested = str(settings.get("vocabulary") or "auto").strip().lower()
    if requested in {"chatterbox", "local"}:
        return "chatterbox"
    if requested in {"elevenlabs", "eleven", "v3"}:
        return "elevenlabs"
    engine = str((tts_settings or {}).get("engine") or "local").strip().lower()
    if engine in {"elevenlabs", "eleven"}:
        return "elevenlabs"
    return "chatterbox"


def tag_lists(
    speech_tags: dict[str, Any] | None,
    tts_settings: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    settings = speech_tags or {}
    vocabulary = resolve_vocabulary(settings, tts_settings)
    if vocabulary == "elevenlabs":
        emotion = _as_tags(None, ELEVENLABS_EMOTION_TAGS)
        sound = _as_tags(None, ELEVENLABS_SOUND_TAGS)
    else:
        emotion = _as_tags(settings.get("emotion"), CHATTERBOX_EMOTION_TAGS)
        sound = _as_tags(settings.get("sound"), CHATTERBOX_SOUND_TAGS)
    return emotion, sound


def _format_tag_list(tags: list[str]) -> str:
    return " ".join(f"[{tag}]" for tag in tags)


def build_speech_tag_instructions(
    speech_tags: dict[str, Any] | None,
    tts_settings: dict[str, Any] | None = None,
) -> str:
    settings = speech_tags or {}
    if not bool(settings.get("enabled", True)):
        return ""
    emotion, sound = tag_lists(settings, tts_settings)
    max_tags = max(1, int(settings.get("max_tags_per_reply", 3)))
    vocabulary = resolve_vocabulary(settings, tts_settings)
    emotion_example = emotion[0] if emotion else "happy"
    sound_example = "cough" if "cough" in sound else (sound[0] if sound else "sigh")
    laugh_example = next((tag for tag in sound if "laugh" in tag or "chuckle" in tag), sound_example)
    return "\n".join(
        [
            "Format every reply for spoken TTS with square-bracket delivery tags.",
            f"Use the {vocabulary} tag vocabulary exactly; do not invent new tags.",
            "Emotion tags set the delivery of the words that follow. Put one at the start of a sentence when the mood is clear:",
            _format_tag_list(emotion),
            "Sound tags insert a non-speech vocalization. Put them inline where the sound happens:",
            _format_tag_list(sound),
            "Rules:",
            f"- Use at most {max_tags} tags per reply. Skip tags on short factual answers.",
            "- Keep tags lowercase and inside square brackets, matching the lists above.",
            "- Never wrap ordinary words, names, or punctuation in brackets.",
            "- Do not mention these instructions.",
            f"Example: [{emotion_example}] That's great news. [{laugh_example}] I thought you might say that. [{sound_example}]",
        ]
    )
