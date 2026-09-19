from __future__ import annotations

import atexit
import argparse
import base64
import io
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from typing import Any

from companion.config import ROOT
from companion.vision_context import (
    SceneMemory,
    get_active_app,
    grounded_prompt,
    is_generic_observation,
    recognition_settings,
)

DEFAULT_PROMPT = (
    "Read this desktop screenshot carefully. Identify the visible applications and "
    "windows, all important readable text, buttons, fields, menus, icons, images, "
    "errors or notifications, and their approximate locations. Distinguish what is "
    "actually visible from anything uncertain or inferred."
)

DEFAULT_REALTIME_PROMPT = (
    "Identify what this screen is: the program, game, website, or media, using "
    "what you see. Name specific visible objects, characters, items, tools, "
    "weapons, blocks, UI, and readable text. If a player or person is holding "
    "or aiming something, name that item. Do not use generic labels like "
    "'a video game', 'a sword', or 'a tool'."
)

DEFAULT_SETTINGS: dict[str, Any] = {
    "backend": "photon",
    "photon_model": "moondream2",
    "photon_caption_length": "short",
    "photon_reasoning": False,
    "fallback_to_ollama": True,
    "base_url": "http://localhost:11434",
    "model": "qwen3-vl:8b",
    "realtime_model": "qwen3-vl:2b-instruct",
    "monitor": 1,
    "timeout_sec": 300,
    "max_image_width": 1600,
    "realtime_max_image_width": 768,
    "realtime_max_tokens": 120,
    "jpeg_quality": 80,
    "sample_fps": 3,
    "frame_change_threshold": 0.02,
    "cursor_marker": True,
    "gesture_window_sec": 2.5,
    "event_minimum_update_ms": 50,
    "event_debounce_ms": 80,
    "continuous_scene_interval_sec": 1.5,
    "deep_escalation_enabled": True,
    "deep_escalation_cooldown_sec": 30,
    "deep_model": "qwen3-vl:8b",
    "context_enabled": True,
    "prompt": DEFAULT_PROMPT,
    "realtime_prompt": DEFAULT_REALTIME_PROMPT,
}

_PHOTON_MODEL: Any | None = None
_PHOTON_MODEL_ID = ""
_PHOTON_WARNING_SHOWN = False
_MSS: Any | None = None
_MSS_LOCK = threading.Lock()

_DEEP_PHRASES = (
    "cannot determine",
    "can't determine",
    "cannot identify",
    "can't identify",
    "not visible",
    "difficult to tell",
    "unable to see",
)


class VisionError(RuntimeError):
    pass


def load_vision_settings() -> dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    for filename in ("configure.json", "config.json"):
        path = ROOT / filename
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VisionError(f"Could not read {path.name}: {exc}") from exc
        section = data.get("screen_visualization")
        if isinstance(section, dict):
            settings.update(section)
            break
    return settings


def _mss():
    global _MSS
    try:
        import mss
    except ImportError as exc:
        raise VisionError("Screen capture needs mss. Run: pip install mss") from exc
    with _MSS_LOCK:
        if _MSS is None:
            _MSS = mss.mss()
        return _MSS


def list_monitors() -> list[str]:
    capture = _mss()
    return [
        (
            f"{index}: {monitor['width']}x{monitor['height']} "
            f"at ({monitor['left']}, {monitor['top']})"
            + (" [all monitors]" if index == 0 else "")
        )
        for index, monitor in enumerate(capture.monitors)
    ]


def _resize_image(image: Any, max_width: int) -> Any:
    from PIL import Image

    if max_width <= 0 or image.width <= max_width:
        return image
    ratio = max_width / image.width
    return image.resize(
        (max_width, max(1, int(image.height * ratio))),
        Image.Resampling.BILINEAR,
    )


def capture_screen(monitor_index: int, max_width: int) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise VisionError(
            "Screen capture needs Pillow. Run: pip install Pillow"
        ) from exc

    capture = _mss()
    monitors = capture.monitors
    if monitor_index < 0 or monitor_index >= len(monitors):
        raise VisionError(
            f"Monitor {monitor_index} does not exist. "
            f"Choose 0-{len(monitors) - 1}."
        )
    shot = capture.grab(monitors[monitor_index])
    image = Image.frombytes("RGB", shot.size, shot.rgb)
    return _resize_image(image, max_width)


def _image_jpeg_bytes(image: Any, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=max(40, min(95, quality)))
    return buffer.getvalue()


def _as_pil(image: Any) -> Any:
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    raise VisionError("Unsupported image type for analysis.")


@dataclass
class MouseSample:
    timestamp: float
    cursor: tuple[float, float]
    left_down: bool


class EventDrivenMonitor:
    """Capture compositor frames and mouse events; only keep the latest RGB keyframe."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self.monitor_index = int(settings.get("monitor", 1))
        self.minimum_update_ms = max(
            16, int(settings.get("event_minimum_update_ms", 50))
        )
        self.gesture_window_sec = max(
            0.5, float(settings.get("gesture_window_sec", 2.5))
        )
        self.max_width = max(320, int(settings.get("realtime_max_image_width", 768)))
        self._lock = threading.Lock()
        self._mouse_samples: deque[MouseSample] = deque(maxlen=4096)
        self._left_down = False
        self._frame_version = 0
        self._last_frame_event = 0.0
        self._last_mouse_event = 0.0
        self._latest_rgb: Any | None = None
        self._latest_preview: Any | None = None
        self._capture_control: Any | None = None
        self._mouse_listener: Any | None = None
        self.error: Exception | None = None
        self._monitor = self._monitor_geometry()

    def monitor_geometry(self) -> dict[str, int]:
        return dict(self._monitor)

    def _monitor_geometry(self) -> dict[str, int]:
        return _monitor_dict(self.monitor_index)

    def start(self) -> None:
        try:
            import cv2
            from pynput import mouse
            from windows_capture import WindowsCapture
        except ImportError as exc:
            raise VisionError(
                "Event-driven monitoring needs windows-capture and pynput. "
                "Run: python vision.py --setup-photon"
            ) from exc

        capture = WindowsCapture(
            cursor_capture=False,
            draw_border=False,
            secondary_window=True,
            minimum_update_interval=self.minimum_update_ms,
            dirty_region=True,
            monitor_index=self.monitor_index,
        )

        @capture.event
        def on_frame_arrived(frame: Any, _control: Any) -> None:
            try:
                bgra = frame.frame_buffer
                height, width = int(frame.height), int(frame.width)
                if self.max_width > 0 and width > self.max_width:
                    new_height = max(1, int(height * self.max_width / width))
                    resized = cv2.resize(
                        bgra,
                        (self.max_width, new_height),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    resized = bgra
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
                gray = cv2.cvtColor(resized, cv2.COLOR_BGRA2GRAY)
                preview = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            except Exception as exc:
                self.error = VisionError(f"Frame capture failed: {exc}")
                return
            with self._lock:
                self._latest_rgb = rgb
                self._latest_preview = preview
                self._frame_version += 1
                self._last_frame_event = time.monotonic()

        @capture.event
        def on_closed() -> None:
            self.error = VisionError("Windows capture session closed.")

        self._capture_control = capture.start_free_threaded()
        self._mouse_listener = mouse.Listener(
            on_move=self._on_mouse_move,
            on_click=self._on_mouse_click,
        )
        self._mouse_listener.start()

    def stop(self) -> None:
        if self._mouse_listener is not None:
            self._mouse_listener.stop()
            self._mouse_listener = None
        if self._capture_control is not None:
            try:
                self._capture_control.stop()
                self._capture_control.wait()
            except Exception:
                pass
            self._capture_control = None

    def snapshot(self) -> tuple[Any | None, Any | None, int, float, float]:
        with self._lock:
            return (
                self._latest_rgb,
                self._latest_preview,
                self._frame_version,
                self._last_frame_event,
                self._last_mouse_event,
            )

    def state(self) -> tuple[int, float, float]:
        with self._lock:
            return (
                self._frame_version,
                self._last_frame_event,
                self._last_mouse_event,
            )

    def cursor(self) -> tuple[float, float] | None:
        with self._lock:
            if self._mouse_samples:
                return self._mouse_samples[-1].cursor
        try:
            from pynput.mouse import Controller

            position = Controller().position
            return self._normalize(int(position[0]), int(position[1]))
        except Exception:
            return None

    def left_down(self) -> bool:
        return self._left_down

    def gesture(self) -> str:
        cutoff = time.monotonic() - self.gesture_window_sec
        with self._lock:
            samples = [
                sample for sample in self._mouse_samples if sample.timestamp >= cutoff
            ]
        points = [sample.cursor for sample in samples]
        return _detect_gesture(samples, points)

    def _normalize(self, x: int, y: int) -> tuple[float, float] | None:
        width = max(1, int(self._monitor["width"]))
        height = max(1, int(self._monitor["height"]))
        normalized = (
            (x - int(self._monitor["left"])) / width,
            (y - int(self._monitor["top"])) / height,
        )
        if 0.0 <= normalized[0] <= 1.0 and 0.0 <= normalized[1] <= 1.0:
            return normalized
        return None

    def _record_mouse(self, x: int, y: int) -> None:
        position = self._normalize(x, y)
        if position is None:
            return
        now = time.monotonic()
        with self._lock:
            self._last_mouse_event = now
            self._mouse_samples.append(
                MouseSample(
                    timestamp=now,
                    cursor=position,
                    left_down=self._left_down,
                )
            )

    def _on_mouse_move(self, x: int, y: int) -> None:
        self._record_mouse(x, y)

    def _on_mouse_click(self, x: int, y: int, button: Any, pressed: bool) -> None:
        try:
            from pynput.mouse import Button

            if button == Button.left:
                self._left_down = pressed
        finally:
            self._record_mouse(x, y)


def _detect_gesture(
    samples: list[MouseSample],
    points: list[tuple[float, float]],
) -> str:
    if len(points) < 4:
        return ""
    drag_points = [
        sample.cursor for sample in samples if sample.left_down
    ]
    if len(drag_points) >= 3:
        start = drag_points[0]
        end = drag_points[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if abs(dx) >= 0.08 and abs(dy) <= 0.06:
            return (
                "The user dragged horizontally, likely selecting or highlighting "
                f"content from ({start[0]:.0%}, {start[1]:.0%}) to "
                f"({end[0]:.0%}, {end[1]:.0%})."
            )
        if math.hypot(dx, dy) >= 0.08:
            return (
                "The user dragged from "
                f"({start[0]:.0%}, {start[1]:.0%}) to "
                f"({end[0]:.0%}, {end[1]:.0%})."
            )

    path_length = sum(
        math.dist(first, second) for first, second in zip(points, points[1:])
    )
    center = (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )
    radii = [math.dist(point, center) for point in points]
    mean_radius = sum(radii) / len(radii)
    angles = [math.atan2(point[1] - center[1], point[0] - center[0]) for point in points]
    unwrapped = [angles[0]]
    for angle in angles[1:]:
        delta = angle - unwrapped[-1]
        while delta > math.pi:
            angle -= 2 * math.pi
            delta = angle - unwrapped[-1]
        while delta < -math.pi:
            angle += 2 * math.pi
            delta = angle - unwrapped[-1]
        unwrapped.append(angle)
    angular_travel = abs(unwrapped[-1] - unwrapped[0])
    closes = math.dist(points[0], points[-1]) <= max(0.05, mean_radius)
    radius_variation = (
        sum(abs(radius - mean_radius) for radius in radii) / len(radii)
        if mean_radius
        else 1.0
    )
    if (
        path_length >= 0.35
        and mean_radius >= 0.025
        and angular_travel >= 4.5
        and closes
        and radius_variation <= mean_radius * 0.75
    ):
        return (
            "The cursor traced a circle around the region centered near "
            f"({center[0]:.0%}, {center[1]:.0%})."
        )

    tail = points[-min(len(points), 8) :]
    if max(math.dist(tail[0], point) for point in tail) <= 0.012:
        return (
            "The cursor is dwelling or pointing near "
            f"({tail[-1][0]:.0%}, {tail[-1][1]:.0%})."
        )
    return ""


def _close_photon() -> None:
    global _PHOTON_MODEL
    if _PHOTON_MODEL is not None:
        try:
            _PHOTON_MODEL.close()
        except Exception:
            pass
        _PHOTON_MODEL = None


def _get_photon(settings: dict[str, Any]) -> Any:
    global _PHOTON_MODEL, _PHOTON_MODEL_ID
    try:
        import moondream as md
    except ImportError as exc:
        raise VisionError(
            "Photon needs the isolated vision environment. "
            "Run: python vision.py --setup-photon"
        ) from exc

    model_id = str(settings.get("photon_model") or "moondream2")
    if _PHOTON_MODEL is None or _PHOTON_MODEL_ID != model_id:
        _close_photon()
        _PHOTON_MODEL = md.photon(model_id)
        _PHOTON_MODEL_ID = model_id
    return _PHOTON_MODEL


def _photon_text(result: Any, key: str) -> str:
    value = result.get(key) if isinstance(result, dict) else None
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return "".join(str(chunk) for chunk in value).strip()


def _overlay_cursor(image: Any, cursor: tuple[float, float]) -> Any:
    from PIL import ImageDraw

    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    x = int(cursor[0] * (marked.width - 1))
    y = int(cursor[1] * (marked.height - 1))
    radius = max(8, min(marked.width, marked.height) // 70)
    box = (x - radius, y - radius, x + radius, y + radius)
    draw.ellipse(box, outline=(255, 0, 255), width=3)
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 255, 0))
    draw.line((x - radius * 2, y, x + radius * 2, y), fill=(255, 255, 0), width=1)
    draw.line((x, y - radius * 2, x, y + radius * 2), fill=(255, 255, 0), width=1)
    return marked


def _frame_change(previous: Any | None, current: Any | None) -> float:
    if previous is None or current is None:
        return 1.0
    try:
        import cv2
    except ImportError:
        return 1.0
    if previous.shape != current.shape:
        return 1.0
    return float(cv2.absdiff(previous, current).mean()) / 255.0


def _analyze_photon(
    settings: dict[str, Any],
    prompt: str,
    image: Any,
    *,
    max_tokens: int | None = None,
    caption: bool = False,
    cursor: tuple[float, float] | None = None,
) -> str:
    from PIL import Image as PILImage

    model = _get_photon(settings)
    pil_image = image if isinstance(image, PILImage.Image) else _as_pil(image)
    sampling = {
        "max_tokens": int(max_tokens or 256),
        "temperature": 0.0,
    }
    reasoning = bool(settings.get("photon_reasoning", False))
    if caption and hasattr(model, "supports") and model.supports("caption"):
        length = str(settings.get("photon_caption_length") or "short")
        answer = _photon_text(
            model.caption(pil_image, length=length, settings=sampling),
            "caption",
        )
    else:
        kwargs: dict[str, Any] = {
            "reasoning": reasoning,
            "settings": sampling,
        }
        if cursor is not None:
            kwargs["spatial_refs"] = [[cursor[0], cursor[1]]]
        try:
            answer = _photon_text(model.query(pil_image, prompt, **kwargs), "answer")
        except TypeError:
            kwargs.pop("spatial_refs", None)
            try:
                answer = _photon_text(model.query(pil_image, prompt, **kwargs), "answer")
            except TypeError:
                answer = _photon_text(model.query(pil_image, prompt), "answer")
    if not answer:
        raise VisionError("Photon returned an empty response.")
    return answer


atexit.register(_close_photon)


def analyze_image(
    settings: dict[str, Any],
    prompt: str,
    image: Any,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    caption: bool = False,
    cursor: tuple[float, float] | None = None,
) -> str:
    global _PHOTON_WARNING_SHOWN
    backend = str(settings.get("backend") or "photon").strip().lower()
    if backend == "photon":
        try:
            return _analyze_photon(
                settings,
                prompt,
                image,
                max_tokens=max_tokens,
                caption=caption,
                cursor=cursor,
            )
        except Exception as exc:
            if not bool(settings.get("fallback_to_ollama", True)):
                if isinstance(exc, VisionError):
                    raise
                raise VisionError(f"Photon failed: {exc}") from exc
            if not _PHOTON_WARNING_SHOWN:
                print(
                    f"Photon unavailable ({exc}); falling back to Ollama.",
                    file=sys.stderr,
                )
                _PHOTON_WARNING_SHOWN = True
    elif backend != "ollama":
        raise VisionError(
            f"Unknown screen_visualization.backend {backend!r}; "
            "use 'photon' or 'ollama'."
        )

    jpeg = (
        image
        if isinstance(image, (bytes, bytearray))
        else _image_jpeg_bytes(image, int(settings.get("jpeg_quality", 80)))
    )
    payload = {
        "model": model or str(settings.get("model") or "qwen3-vl:8b"),
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [base64.b64encode(jpeg).decode("ascii")],
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": max_tokens or 512,
        },
    }
    base_url = str(settings.get("base_url") or "http://localhost:11434").rstrip("/")
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=int(settings.get("timeout_sec", 300)),
        ) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise VisionError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise VisionError(
            f"Could not reach Ollama at {base_url}: {exc.reason}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise VisionError("Ollama returned invalid JSON.") from exc

    message = result.get("message") if isinstance(result, dict) else None
    text = message.get("content") if isinstance(message, dict) else None
    if not text:
        raise VisionError("The vision model returned an empty response.")
    return str(text).strip()


def analyze_screen(settings: dict[str, Any], prompt: str) -> str:
    monitor_index = int(settings.get("monitor", 1))
    image = capture_screen(
        monitor_index,
        int(settings.get("max_image_width", 1600)),
    )
    if bool(settings.get("context_enabled", True)):
        app = get_active_app(_monitor_dict(monitor_index), settings)
        prompt = grounded_prompt(base_prompt=prompt, app=app, richer=True)
    return analyze_image(
        recognition_settings(settings),
        prompt,
        image,
        model=str(settings.get("model") or "qwen3-vl:8b"),
        max_tokens=int(settings.get("max_tokens", 256)),
    )


def _monitor_dict(monitor_index: int) -> dict[str, int]:
    capture = _mss()
    if monitor_index < 0 or monitor_index >= len(capture.monitors):
        raise VisionError(
            f"Monitor {monitor_index} does not exist. "
            f"Choose 0-{len(capture.monitors) - 1}."
        )
    return dict(capture.monitors[monitor_index])


def _needs_deep_analysis(result: str) -> bool:
    lowered = result.lower()
    return is_generic_observation(result) or any(
        phrase in lowered for phrase in _DEEP_PHRASES
    )


def _run_deep_analysis(
    settings: dict[str, Any],
    image: Any,
    prompt: str,
) -> None:
    deep_settings = dict(settings)
    deep_settings["backend"] = "ollama"
    model = str(settings.get("deep_model") or "qwen3-vl:8b")
    deep_prompt = (
        prompt
        + " Identify the specific application, game, show, artwork, or real-world "
        "subject. Use proper names and relevant cultural or gameplay context."
    )
    try:
        result = analyze_image(
            deep_settings,
            deep_prompt,
            image,
            model=model,
            max_tokens=512,
        )
    except Exception as exc:
        print(
            f"\n[deep analysis unavailable: {exc}]",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(f"\n[deep {model}] {result}", flush=True)


def _watch_prompt(
    settings: dict[str, Any],
    cursor: tuple[float, float] | None,
    gesture: str,
) -> str:
    prompt = str(settings.get("realtime_prompt") or DEFAULT_REALTIME_PROMPT)
    extras: list[str] = []
    if cursor is not None:
        extras.append(
            f"Cursor coordinates are ({cursor[0]:.2f}, {cursor[1]:.2f}) "
            "normalized to the image. A magenta circle marks that point."
        )
    if gesture:
        extras.append("Direct mouse-event observation: " + gesture)
    if extras:
        prompt += " " + " ".join(extras)
    return prompt


def watch_screen(settings: dict[str, Any], interval: float | None = None) -> None:
    from PIL import Image

    monitor = int(settings.get("monitor", 1))
    sample_fps = max(1.0, float(settings.get("sample_fps", 3)))
    min_interval = 1.0 / sample_fps
    if interval is not None:
        min_interval = min(min_interval, max(0.08, float(interval)))
    heartbeat = max(min_interval, float(settings.get("continuous_scene_interval_sec", 1.5)))
    change_threshold = max(0.005, float(settings.get("frame_change_threshold", 0.02)))
    deep_enabled = bool(settings.get("deep_escalation_enabled", True))
    deep_cooldown = float(settings.get("deep_escalation_cooldown_sec", 30))
    recognize = recognition_settings(settings)
    model = str(settings.get("realtime_model") or "qwen3-vl:2b-instruct")
    max_tokens = int(settings.get("realtime_max_tokens", 120))
    mark_cursor = bool(settings.get("cursor_marker", True))
    context_enabled = bool(settings.get("context_enabled", True))
    events = EventDrivenMonitor(settings)
    events.start()
    memory = SceneMemory()
    last_analysis = 0.0
    last_version = -1
    last_gesture = ""
    last_cursor: tuple[float, float] | None = None
    last_preview: Any | None = None
    last_deep_analysis = 0.0
    last_significant_change = 0.0
    deep_thread: threading.Thread | None = None
    print(
        f"Realtime vision on monitor {monitor} with {model} "
        f"(cap {sample_fps:.0f} FPS, skip < {change_threshold:.0%} change). "
        "The VL model names the app and objects from the image; "
        "Windows only supplies the live window title. "
        "Press Ctrl+C to stop.",
        flush=True,
    )
    try:
        while True:
            if events.error is not None:
                raise VisionError(f"Event monitoring failed: {events.error}")
            deep_busy = deep_thread is not None and deep_thread.is_alive()
            if deep_busy:
                time.sleep(0.05)
                continue
            now = time.monotonic()
            rgb, preview, version, _last_frame_event, _last_mouse_event = events.snapshot()
            if rgb is None:
                time.sleep(0.02)
                continue
            cursor = events.cursor()
            gesture = events.gesture()
            gesture_changed = bool(gesture and gesture != last_gesture)
            change = _frame_change(last_preview, preview)
            significant = change >= change_threshold
            if significant:
                last_significant_change = now
            mouse_interesting = bool(gesture_changed or events.left_down())
            app = (
                get_active_app(events.monitor_geometry(), settings)
                if context_enabled
                else None
            )
            app_changed = bool(app and memory.app_changed(app))
            rate_ok = last_analysis == 0.0 or now - last_analysis >= min_interval
            heartbeat_due = (
                version != last_version
                and now - last_analysis >= heartbeat
            )
            should_analyze = rate_ok and (
                last_analysis == 0.0
                or mouse_interesting
                or significant
                or heartbeat_due
                or app_changed
            )
            if not should_analyze:
                time.sleep(0.016)
                continue

            image = Image.fromarray(rgb.copy(), "RGB")
            if mark_cursor and cursor is not None:
                image = _overlay_cursor(image, cursor)
            if app is not None:
                prompt = grounded_prompt(
                    base_prompt=str(
                        settings.get("realtime_prompt") or DEFAULT_REALTIME_PROMPT
                    ),
                    app=app,
                    cursor=cursor,
                    gesture=gesture,
                    memory=memory,
                    richer=app_changed or last_analysis == 0.0,
                )
            else:
                prompt = _watch_prompt(settings, cursor, gesture)
            started = time.monotonic()
            result = analyze_image(
                recognize,
                prompt,
                image,
                model=model,
                max_tokens=max(max_tokens, 160) if app_changed else max_tokens,
                caption=False,
                cursor=cursor if mouse_interesting else None,
            )
            elapsed = time.monotonic() - started
            app_bit = f", {app.label()}" if app is not None else ""
            print(
                f"\n[{elapsed:.2f}s{app_bit}, {change:.0%} change, event {version}"
                + (f", gesture: {gesture}" if gesture_changed else "")
                + (f", cursor ({cursor[0]:.0%}, {cursor[1]:.0%})" if cursor else "")
                + f"] {result}",
                flush=True,
            )
            now = time.monotonic()
            screen_quiet = now - last_significant_change >= 1.0
            if (
                deep_enabled
                and (screen_quiet or app_changed)
                and _needs_deep_analysis(result)
                and now - last_deep_analysis >= deep_cooldown
                and (deep_thread is None or not deep_thread.is_alive())
            ):
                deep_thread = threading.Thread(
                    target=_run_deep_analysis,
                    args=(settings, image, prompt),
                    daemon=True,
                )
                deep_thread.start()
                last_deep_analysis = now
            if app is not None:
                memory.remember(app, result)
            last_gesture = gesture
            last_cursor = cursor
            time.sleep(0.04)
            _, preview_after, version_after, _, _ = events.snapshot()
            last_preview = preview_after if preview_after is not None else preview
            last_version = version_after
            last_analysis = time.monotonic()
    finally:
        events.stop()


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze the current desktop with a local vision model"
    )
    parser.add_argument("--monitor", type=int, help="Monitor number; 0 captures all")
    parser.add_argument("--model", help="Ollama vision model override")
    parser.add_argument("--prompt", help="Analyze once with this question and exit")
    parser.add_argument(
        "--watch",
        type=float,
        nargs="?",
        const=0.12,
        metavar="SECONDS",
        help="Watch the screen; optional minimum sample interval",
    )
    parser.add_argument("--list-monitors", action="store_true")
    args = parser.parse_args(argv)

    try:
        settings = load_vision_settings()
        if args.monitor is not None:
            settings["monitor"] = args.monitor
        if args.model:
            settings["model"] = args.model
        if args.list_monitors:
            print("Available screens:")
            for monitor in list_monitors():
                print(f"  {monitor}")
            return 0

        default_prompt = str(settings.get("prompt") or DEFAULT_PROMPT)
        if args.prompt:
            print(analyze_screen(settings, args.prompt))
            return 0
        if args.watch is not None:
            watch_screen(settings, args.watch)
            return 0

        print(
            "OpenCompanion screen vision - "
            f"{settings.get('photon_model') or settings['model']} "
            f"on monitor {settings['monitor']}"
        )
        print("Enter a question, press Enter for a full description, or type /exit.")
        while True:
            try:
                prompt = input("\nScreen> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nStopped.")
                return 0
            if prompt.lower() in {"/exit", "/quit"}:
                return 0
            print("\nVision> " + analyze_screen(settings, prompt or default_prompt))
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0
    except (OSError, ValueError, VisionError) as exc:
        print(f"Screen vision failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run())
