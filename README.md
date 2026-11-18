# MotionTimeControl

Control a video timeline with your hand: spread your thumb and index finger to move forward, pinch to rewind, and remove your hand to hold the frame.

[Русская инструкция](README.ru.md) · [License](LICENSE) · [Third-party components](THIRD_PARTY.md)

A local Python/OpenCV prototype using MediaPipe hand landmarks. No API key, cloud inference, account, or model training is required. Camera frames are processed in memory; the application does not record or upload them.

## Features

- **1 — flower, 2 — rocket**, with shared calibration and timeline position.
- Two-step calibration, distance normalization, smoothing, and automatic hold when the hand disappears.
- Manual scrubbing with the mouse, freeze, fullscreen, and a clean video-only view.
- Original procedural demo clips generated locally during setup. Real footage is optional and is not distributed in this repository.

## Quick start

Use **Python 3.12** and a desktop session with a webcam. The pinned dependencies and live camera flow were tested on Apple Silicon / macOS 15+. Linux and Windows commands are provided, but live camera behavior on those systems has not been manually validated. A headless server cannot display the desktop window.

```sh
git clone https://github.com/satkan-satkan/MotionTimeControl.git
cd MotionTimeControl
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python setup_assets.py
python app.py
```

On Windows, use `py -3.12 -m venv .venv`, then `.venv\Scripts\Activate.ps1` in PowerShell instead of the environment commands above. If activation is restricted, call `.venv\Scripts\python.exe` directly.

On Ubuntu/Debian, install the native graphics libraries before running the model (MediaPipe needs them even with CPU inference):

```sh
sudo apt-get update
sudo apt-get install -y libegl1 libgl1
```

On macOS, `./setup.command` and then `./run.command` provide the same setup/launch flow. Allow camera access for the launching application (usually Terminal). After granting permission, press **R** or restart the app.

Setup downloads Google's official Hand Landmarker model (about 8 MB) and verifies its SHA-256. Subsequent runs work offline. Existing local videos are not overwritten.

### Without a camera or model download

After installing requirements:

```sh
python setup_assets.py --demo-only
python app.py --no-camera
```

## Controls

Show one hand, palm facing the camera, in good light. Pinch your thumb and index finger and press **C**; spread them to a comfortable maximum and press **C** again. Keep the palm at a similar angle while controlling the timeline.

| Key / action | Result |
| --- | --- |
| **1 / 2** or top tabs | Flower / rocket |
| **C** | Calibrate closed, then open |
| **X** | Reset or cancel calibration |
| **Space** | Freeze / resume hand control |
| **M** | Manual / hand control |
| Drag the timeline | Scrub manually; press M to return to hand control |
| **H** | Clean video-only view |
| **F** | Fullscreen |
| **R** | Reconnect the camera |
| **Esc / Q** | Quit and release the camera |

Use the English keyboard layout for letter shortcuts. Switching clips preserves calibration, relative position, manual mode and freeze state. Calibration currently resets when the application restarts.

```sh
python app.py --clip rocket
python app.py --camera 1
python app.py --video "my-clip.mp4"  # add a custom clip on key 3
```

## Use real footage

Place your own short clips at `assets/flower.mp4` and/or `assets/rocket.mp4`. They take precedence over generated demos. For the source clips used during development, see [THIRD_PARTY.md](THIRD_PARTY.md). Those videos are not automatically downloaded.

With FFmpeg installed, prepare the same intervals from local source files:

```sh
ffmpeg -ss 76 -i "flower-source.mp4" -t 7 -an -vf "fps=24:start_time=0,scale=1280:-2" -c:v libx264 -crf 18 -movflags +faststart assets/flower.mp4
ffmpeg -ss 18 -i "rocket-source.mp4" -t 7 -an -vf "fps=30:start_time=0,scale=1280:-2" -c:v libx264 -crf 18 -movflags +faststart assets/rocket.mp4
```

Use material you have permission to use. The app is silent. Prefer constant frame rate clips, under 30 seconds / 900 frames. Frames are cached as JPEG in memory, with a 192 MiB limit per clip, so forward/backward jumps do not repeatedly seek through the original video. The default labels describe the development clips; use `--video` for arbitrary footage so it is labeled CUSTOM.

## Checks

```sh
python -m unittest discover -s tests -v
python app.py --check
python app.py --clip rocket --snapshot previews/rocket.png
python app.py --camera-check
```

Tests generate their own fixtures and do not use a webcam. `--check` checks both clips and real model inference on blank frames. `--camera-check` separately reads the webcam for five seconds and prints detection counts without saving frames. CI checks installation, demo creation and model inference without opening a desktop window or camera.

## Privacy and publication

- No telemetry, camera recording, or network calls in the application runtime.
- Setup contacts PyPI and Google's model storage. GitHub Actions runs only on repository contents.
- Videos, model binaries, screenshots, previews, logs, virtual environments, `.env` files and local editor settings are ignored by Git.
- Generated demos contain no personal images and can be recreated from source code.
- Before sharing diagnostics, inspect them for local filesystem paths or other personal details.

## Project layout

`app.py` handles the window, clips and detector. `control.py` implements gesture mapping and smoothing. `demo_assets.py` creates demo animations. `setup_assets.py` verifies the model download. `tests/` exercises gesture behavior and setup/clip switching.

## Possible next steps

- Save calibration between launches and offer guided on-screen calibration.
- Add a clip picker and trim controls, so changing a video needs no terminal.
- Package a macOS app with a clear camera permission flow.
- Add adjustable smoothing and a confidence indicator for difficult lighting.
- Add optional vertical demo recording with explicit camera-preview controls.

## License

Project code and original demo animation code: **MIT**. Dependencies, model files and user-supplied media retain their own terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
