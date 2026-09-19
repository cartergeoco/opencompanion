from __future__ import annotations

import argparse
import sys
from datetime import datetime

from companion.config import load_config
from companion.listen import VoiceDetector, list_input_devices
from companion.speak import SpeakError, Speaker


def _meter(energy: float, threshold: float, width: int = 20) -> str:
    filled = min(width, int((energy / max(threshold, 1e-6)) * (width / 2)))
    return "#" * filled + "-" * (width - filled)


def listen() -> int:
    settings = load_config()
    detector = VoiceDetector(
        sample_rate=int(settings["sample_rate"]),
        block_ms=int(settings["block_ms"]),
        speech_threshold=float(settings["speech_threshold"]),
        start_frames=int(settings["start_frames"]),
        end_frames=int(settings["end_frames"]),
        device=settings["device"],
    )
    try:
        speaker = Speaker(settings) if settings.get("speak_aloud") else None
    except SpeakError as exc:
        print(f"Could not start TTS: {exc}", file=sys.stderr)
        return 1
    was_speaking = False

    print("OpenCompanion is listening. Speak into the microphone.")
    if speaker is not None:
        print(f"TTS: {speaker.label}")
    print("Press Ctrl+C to stop.")
    print()

    try:
        detector.open()
        while True:
            energy, speaking = detector.read()
            status = "talking" if speaking else "quiet  "
            line = f"[{_meter(energy, detector.speech_threshold)}] {status}  energy={energy:.4f}"
            print(f"\r{line:<72}", end="", flush=True)

            if speaking and not was_speaking:
                stamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{stamp}] I hear you talking.")
                if speaker is not None:
                    speaker.say("I hear you talking.")
                    detector.drain()
                    speaking = False
            was_speaking = speaking
    except KeyboardInterrupt:
        print("\nStopped listening.")
        return 0
    except Exception as exc:
        print(f"\nCould not use the microphone: {exc}", file=sys.stderr)
        return 1
    finally:
        detector.close()


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenCompanion voice listener")
    parser.add_argument(
        "--devices",
        action="store_true",
        help="List microphones and exit",
    )
    args = parser.parse_args(argv)
    if args.devices:
        print("Input devices (> marks the default):")
        for line in list_input_devices():
            print(line)
        return 0
    return listen()
