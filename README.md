# OpenCompanion

A file-based voice companion with no frontend. Speak into the microphone and it acknowledges when it hears you talking.

## Branches

- `main` — stable releases
- `nightly` — current development

Publishing a GitHub Release (not a prerelease) fast-forwards `main` to that tag.

## Run

```powershell
pip install -r requirements.txt
python run.py
```

List microphones:

```powershell
python run.py --devices
```

Adjust `config.json` if it misses speech or reacts to room noise.
