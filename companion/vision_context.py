from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GENERIC_PHRASES = (
    "a computer screen",
    "a screenshot",
    "an image of a screen",
    "a video game",
    "a desktop",
    "some kind of game",
    "a window showing",
    "the screen shows",
    "an application window",
    "holding a tool",
    "holding a sword",
    "holding a weapon",
    "a blocky",
    "pixelated game",
)


@dataclass
class ForegroundApp:
    title: str = ""
    exe: str = ""
    pid: int = 0

    @property
    def key(self) -> str:
        return f"{self.exe}|{self.title}"

    def label(self) -> str:
        if self.title:
            return self.title[:80]
        return self.exe or "unknown window"


class SceneMemory:
    def __init__(self, max_chars: int = 240) -> None:
        self.max_chars = max_chars
        self.app_key = ""
        self.last_text = ""

    def app_changed(self, app: ForegroundApp) -> bool:
        return bool(self.app_key) and app.key != self.app_key

    def remember(self, app: ForegroundApp, text: str) -> None:
        self.app_key = app.key
        self.last_text = " ".join(text.split())[: self.max_chars]


_WIN32: tuple[Any, Any, Any] | None | bool = False


def _load_win32() -> tuple[Any, Any, Any] | None:
    global _WIN32
    if _WIN32 is False:
        if sys.platform != "win32":
            _WIN32 = None
            return None
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        dwmapi = ctypes.windll.dwmapi
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
        user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
        user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        user32.IsWindowVisible.argtypes = [wintypes.HWND]
        user32.IsIconic.argtypes = [wintypes.HWND]
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        dwmapi.DwmGetWindowAttribute.argtypes = [
            wintypes.HWND,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        _WIN32 = (user32, kernel32, dwmapi)
    return None if _WIN32 is None else _WIN32


def _window_title(user32: Any, hwnd: int) -> str:
    length = int(user32.GetWindowTextLengthW(hwnd)) + 1
    if length <= 1:
        return ""
    buffer = ctypes.create_unicode_buffer(length)
    user32.GetWindowTextW(hwnd, buffer, length)
    return buffer.value.strip()


def _window_rect(user32: Any, hwnd: int) -> tuple[int, int, int, int] | None:
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    if rect.right <= rect.left or rect.bottom <= rect.top:
        return None
    return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))


def _overlap_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0, right - left) * max(0, bottom - top)


def _is_cloaked(dwmapi: Any, hwnd: int) -> bool:
    cloaked = wintypes.DWORD()
    try:
        status = dwmapi.DwmGetWindowAttribute(
            hwnd,
            14,  # DWMWA_CLOAKED
            ctypes.byref(cloaked),
            ctypes.sizeof(cloaked),
        )
    except Exception:
        return False
    return status == 0 and cloaked.value != 0


def _process_exe(kernel32: Any, user32: Any, hwnd: int) -> tuple[int, str]:
    pid = wintypes.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        return 0, ""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return int(pid.value), ""
    try:
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return int(pid.value), Path(buffer.value).name.lower()
    finally:
        kernel32.CloseHandle(handle)
    return int(pid.value), ""


def _visible_top_level(user32: Any, dwmapi: Any, hwnd: int) -> bool:
    if not hwnd or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
        return False
    return not _is_cloaked(dwmapi, hwnd)


def _enum_windows(user32: Any) -> list[int]:
    hwnds: list[int] = []
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _callback(hwnd: int, _lparam: int) -> bool:
        hwnds.append(int(hwnd))
        return True

    callback = proto(_callback)
    user32.EnumWindows(callback, 0)
    return hwnds


def _pick_hwnd(
    user32: Any,
    dwmapi: Any,
    monitor: tuple[int, int, int, int] | None,
) -> int:
    foreground = int(user32.GetForegroundWindow() or 0)
    if foreground and _visible_top_level(user32, dwmapi, foreground):
        rect = _window_rect(user32, foreground)
        if rect is None or monitor is None:
            return foreground
        monitor_area = max(1, (monitor[2] - monitor[0]) * (monitor[3] - monitor[1]))
        if _overlap_area(rect, monitor) / monitor_area >= 0.08:
            return foreground

    if monitor is None:
        return foreground

    best = 0
    best_area = 0
    for hwnd in _enum_windows(user32):
        if not _visible_top_level(user32, dwmapi, hwnd):
            continue
        rect = _window_rect(user32, hwnd)
        if rect is None:
            continue
        area = _overlap_area(rect, monitor)
        if area > best_area:
            best = hwnd
            best_area = area
    return best or foreground


def monitor_rect(monitor: dict[str, int] | None) -> tuple[int, int, int, int] | None:
    if not monitor:
        return None
    left = int(monitor.get("left", 0))
    top = int(monitor.get("top", 0))
    width = int(monitor.get("width", 0))
    height = int(monitor.get("height", 0))
    if width <= 0 or height <= 0:
        return None
    return (left, top, left + width, top + height)


def get_active_app(
    monitor: dict[str, int] | None = None,
    _settings: dict[str, Any] | None = None,
) -> ForegroundApp:
    loaded = _load_win32()
    if loaded is None:
        return ForegroundApp()
    user32, kernel32, dwmapi = loaded
    try:
        hwnd = _pick_hwnd(user32, dwmapi, monitor_rect(monitor))
        if not hwnd:
            return ForegroundApp()
        title = _window_title(user32, hwnd)
        pid, exe = _process_exe(kernel32, user32, hwnd)
        return ForegroundApp(title=title, exe=exe, pid=pid)
    except Exception:
        return ForegroundApp()


def is_generic_observation(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in GENERIC_PHRASES)


def recognition_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Use the stronger local VL model for naming apps and in-world objects."""
    updated = dict(settings)
    updated["backend"] = "ollama"
    return updated


def grounded_prompt(
    *,
    base_prompt: str,
    app: ForegroundApp,
    cursor: tuple[float, float] | None = None,
    gesture: str = "",
    memory: SceneMemory | None = None,
    richer: bool = False,
) -> str:
    parts: list[str] = [
        "Windows reports these as facts about the focused window, not as a game list:",
    ]
    if app.title:
        parts.append(f'window title "{app.title[:160]}"')
    if app.exe:
        parts.append(f"process {app.exe}")
    parts.append(".")
    parts.append(base_prompt)
    previous = (
        memory.last_text
        if memory and not memory.app_changed(app) and memory.last_text
        else ""
    )
    if previous:
        parts.append(f"Previously: {previous} What changed? Keep specific names.")
    if richer:
        parts.append(
            "Read HUD, hotbar, captions, and item names if they are visible."
        )
    if cursor is not None:
        parts.append(
            f"The magenta mark is the cursor at ({cursor[0]:.2f}, {cursor[1]:.2f}). "
            "Name the object there."
        )
    if gesture:
        parts.append("Mouse observation: " + gesture)
    return " ".join(parts)
