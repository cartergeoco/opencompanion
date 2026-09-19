from __future__ import annotations

import json
import os
import struct
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import sounddevice as sd

from companion.config import ROOT


class SpeakError(RuntimeError):
    """Raised when speech cannot be synthesized or played."""


class SpeakBackend(Protocol):
    label: str

    def say(self, text: str, stop_event: threading.Event | None = None) -> None: ...


def _play_float_audio(
    samples: np.ndarray,
    sample_rate: int,
    stop_event: threading.Event | None = None,
) -> None:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return
    if stop_event is not None and stop_event.is_set():
        return
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / peak
    sd.play(audio, sample_rate)
    if stop_event is None:
        sd.wait()
        return
    while True:
        if stop_event.is_set():
            sd.stop()
            return
        try:
            stream = sd.get_stream()
        except Exception:
            return
        if stream is None or not getattr(stream, "active", False):
            return
        time.sleep(0.03)


def _resolve_device(device: str) -> str:
    requested = (device or "auto").strip().lower()
    if requested and requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _apply_hf_token(token: str) -> None:
    value = token.strip()
    if not value:
        return
    os.environ.setdefault("HF_TOKEN", value)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", value)


class LocalChatterboxSpeaker:
    """Resemble AI Chatterbox Turbo: realistic, expressive, low-latency local TTS."""

    def __init__(self, settings: dict[str, Any], hf_token: str = "") -> None:
        try:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
        except ImportError as exc:
            raise SpeakError(
                "Local TTS needs Chatterbox Turbo. Run: pip install -r requirements-tts.txt"
            ) from exc

        _apply_hf_token(hf_token)
        model_name = str(settings.get("model") or "chatterbox-turbo").lower()
        device = _resolve_device(str(settings.get("device") or "auto"))
        self._model = ChatterboxTurboTTS.from_pretrained(
            device=device,
            nano="nano" in model_name,
        )
        self._temperature = float(settings.get("temperature", 0.8))
        self._top_p = float(settings.get("top_p", 0.95))
        self._repetition_penalty = float(settings.get("repetition_penalty", 1.2))
        prompt = str(settings.get("audio_prompt_path") or "").strip()
        self._audio_prompt_path = ""
        if prompt:
            path = Path(prompt)
            if not path.is_absolute():
                path = ROOT / path
            if not path.exists():
                raise SpeakError(f"TTS voice prompt not found: {path}")
            self._audio_prompt_path = str(path)
        kind = "Nano" if "nano" in model_name else "Turbo"
        self.label = f"local Chatterbox {kind} on {device}"

    def say(self, text: str, stop_event: threading.Event | None = None) -> None:
        spoken = text.strip()
        if not spoken:
            return
        if stop_event is not None and stop_event.is_set():
            return
        kwargs: dict[str, Any] = {
            "temperature": self._temperature,
            "top_p": self._top_p,
            "repetition_penalty": self._repetition_penalty,
        }
        if self._audio_prompt_path:
            kwargs["audio_prompt_path"] = self._audio_prompt_path
        wav = self._model.generate(spoken, **kwargs)
        if stop_event is not None and stop_event.is_set():
            return
        audio = np.asarray(wav.detach().cpu().numpy(), dtype=np.float32)
        _play_float_audio(audio, int(self._model.sr), stop_event)


class ElevenLabsSpeaker:
    """Hosted ElevenLabs TTS."""

    def __init__(self, settings: dict[str, Any]) -> None:
        api_key = str(settings.get("api_key") or os.getenv("ELEVENLABS_API_KEY") or "").strip()
        if not api_key:
            raise SpeakError(
                "ElevenLabs TTS needs tts.elevenlabs.api_key or the ELEVENLABS_API_KEY environment variable."
            )
        voice_id = str(settings.get("voice_id") or "").strip()
        if not voice_id:
            raise SpeakError("ElevenLabs TTS needs tts.elevenlabs.voice_id.")
        base_url = str(settings.get("base_url") or "https://api.elevenlabs.io").rstrip("/")
        self._api_key = api_key
        self._voice_id = voice_id
        self._model_id = str(settings.get("model") or "eleven_v3")
        self._output_format = str(settings.get("output_format") or "pcm_24000")
        self._url = (
            f"{base_url}/v1/text-to-speech/{voice_id}"
            f"?output_format={self._output_format}"
        )
        self._voice_settings = {
            "stability": float(settings.get("stability", 0.45)),
            "similarity_boost": float(settings.get("similarity_boost", 0.75)),
            "style": float(settings.get("style", 0.4)),
            "use_speaker_boost": True,
            "speed": float(settings.get("speed", 1.0)),
        }
        self.label = f"ElevenLabs {self._model_id}"

    def say(self, text: str, stop_event: threading.Event | None = None) -> None:
        spoken = text.strip()
        if not spoken:
            return
        if stop_event is not None and stop_event.is_set():
            return
        payload = json.dumps(
            {
                "text": spoken,
                "model_id": self._model_id,
                "voice_settings": self._voice_settings,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=payload,
            method="POST",
            headers={
                "xi-api-key": self._api_key,
                "Accept": "application/octet-stream",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                audio_bytes = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SpeakError(_elevenlabs_error(exc.code, detail)) from exc
        except urllib.error.URLError as exc:
            raise SpeakError(f"ElevenLabs TTS request failed: {exc}") from exc
        if stop_event is not None and stop_event.is_set():
            return
        sample_rate = _pcm_sample_rate(self._output_format)
        audio = _decode_pcm16(audio_bytes)
        _play_float_audio(audio, sample_rate, stop_event)


class PyttsxSpeaker:
    """Built-in system voice fallback."""

    def __init__(self) -> None:
        import pyttsx3

        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 180)
        self.label = "system pyttsx3"

    def say(self, text: str, stop_event: threading.Event | None = None) -> None:
        spoken = text.strip()
        if not spoken:
            return
        if stop_event is not None and stop_event.is_set():
            return
        self.engine.say(spoken)
        self.engine.runAndWait()


def _elevenlabs_error(status: int, detail: str) -> str:
    lowered = detail.lower()
    if status == 402 or "paid_plan_required" in lowered or "library voices" in lowered:
        return (
            "ElevenLabs blocked this voice on a free API plan. Use a premade "
            "voice such as George (JBFqnCBsd6RMkjVDRZzb), or upgrade to use "
            "library voices."
        )
    snippet = " ".join(detail.split())[:280]
    return f"ElevenLabs TTS failed ({status}): {snippet}"


def _pcm_sample_rate(output_format: str) -> int:
    parts = output_format.lower().split("_")
    for part in parts:
        if part.isdigit():
            return int(part)
    return 24000


def _decode_pcm16(data: bytes) -> np.ndarray:
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return _wav_to_float(data)
    usable = len(data) - (len(data) % 2)
    return np.frombuffer(data[:usable], dtype="<i2").astype(np.float32) / 32768.0


def _wav_to_float(data: bytes) -> np.ndarray:
    if len(data) < 44:
        raise SpeakError("ElevenLabs returned an empty WAV file.")
    channels = struct.unpack_from("<H", data, 22)[0]
    bits = struct.unpack_from("<H", data, 34)[0]
    offset = data.find(b"data")
    if offset < 0 or offset + 8 > len(data):
        raise SpeakError("ElevenLabs WAV response is missing audio data.")
    payload = data[offset + 8 :]
    if bits == 16:
        samples = np.frombuffer(payload[: len(payload) - (len(payload) % 2)], dtype="<i2")
        audio = samples.astype(np.float32) / 32768.0
    elif bits == 32:
        samples = np.frombuffer(payload[: len(payload) - (len(payload) % 4)], dtype="<i4")
        audio = samples.astype(np.float32) / 2147483648.0
    else:
        raise SpeakError(f"Unsupported WAV sample width: {bits} bits")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


def create_backend(settings: dict[str, Any]) -> SpeakBackend:
    tts = settings.get("tts")
    if not isinstance(tts, dict):
        return PyttsxSpeaker()
    engine = str(tts.get("engine") or "local").strip().lower()
    if engine in {"elevenlabs", "eleven"}:
        section = tts.get("elevenlabs")
        return ElevenLabsSpeaker(section if isinstance(section, dict) else {})
    if engine in {"local", "chatterbox", "chatterbox-turbo"}:
        section = tts.get("local")
        return LocalChatterboxSpeaker(
            section if isinstance(section, dict) else {},
            hf_token=str(settings.get("hf_token") or os.getenv("HF_TOKEN") or ""),
        )
    if engine in {"pyttsx3", "system"}:
        return PyttsxSpeaker()
    raise SpeakError(f"Unknown TTS engine {engine!r}. Use local or elevenlabs.")


class Speaker:
    """Speak text with the configured TTS engine."""

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self._backend = create_backend(settings or {})
        self.label = self._backend.label
        self._stop = threading.Event()
        self._busy = threading.Event()
        self._thread: threading.Thread | None = None
        self._current_text = ""
        self._error = ""

    def is_speaking(self) -> bool:
        return self._busy.is_set()

    def current_text(self) -> str:
        return self._current_text

    def take_error(self) -> str:
        error, self._error = self._error, ""
        return error

    def interrupt(self) -> None:
        self._stop.set()
        try:
            sd.stop()
        except Exception:
            pass
        engine = getattr(self._backend, "engine", None)
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.2)
        self._busy.clear()

    def say(self, text: str) -> None:
        self.interrupt()
        stop = threading.Event()
        self._stop = stop
        self._current_text = text
        self._busy.set()
        try:
            self._backend.say(text, stop)
        finally:
            if self._stop is stop:
                self._busy.clear()

    def say_async(self, text: str) -> None:
        self.interrupt()
        stop = threading.Event()
        self._stop = stop
        self._error = ""
        self._current_text = text
        self._busy.set()
        self._thread = threading.Thread(
            target=self._run_async,
            args=(text, stop),
            daemon=True,
        )
        self._thread.start()

    def _run_async(self, text: str, stop: threading.Event) -> None:
        try:
            if not stop.is_set():
                self._backend.say(text, stop)
        except SpeakError as exc:
            if not stop.is_set():
                self._error = str(exc)
        except Exception as exc:
            if not stop.is_set():
                self._error = f"TTS failed: {exc}"
        finally:
            if self._stop is stop:
                self._busy.clear()
