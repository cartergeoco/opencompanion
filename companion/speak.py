from __future__ import annotations

import pyttsx3


class Speaker:
    """Speak a short confirmation out loud."""

    def __init__(self) -> None:
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 180)

    def say(self, text: str) -> None:
        self.engine.say(text)
        self.engine.runAndWait()
