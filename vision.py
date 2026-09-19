from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

from companion.vision_app import run


ROOT = Path(__file__).resolve().parent
VISION_ENV = ROOT / ".venv-vision"
VISION_PYTHON = (
    VISION_ENV / "Scripts" / "python.exe"
    if sys.platform == "win32"
    else VISION_ENV / "bin" / "python"
)


def setup_photon() -> int:
    print(f"Creating isolated vision environment at {VISION_ENV}...")
    venv.EnvBuilder(with_pip=True).create(VISION_ENV)
    completed = subprocess.run(
        [
            str(VISION_PYTHON),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "moondream==2.3.0",
            "mss>=10.0",
            "Pillow>=11.0",
            "windows-capture==2.0.1",
            "pynput>=1.8",
        ],
        check=False,
    )
    if completed.returncode == 0 and sys.platform == "win32":
        completed = subprocess.run(
            [
                str(VISION_PYTHON),
                "-m",
                "pip",
                "install",
                "torch==2.8.0",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
            check=False,
        )
    if completed.returncode == 0:
        print("Photon setup complete. Run: python vision.py --watch")
    return completed.returncode


if __name__ == "__main__":
    if "--setup-photon" in sys.argv[1:]:
        raise SystemExit(setup_photon())
    if VISION_PYTHON.exists() and Path(sys.executable).resolve() != VISION_PYTHON.resolve():
        raise SystemExit(
            subprocess.call([str(VISION_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
        )
    raise SystemExit(run())
