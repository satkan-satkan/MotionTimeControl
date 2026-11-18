"""Time Control: local, hand-controlled flower and rocket timelines. No camera uploads."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import cv2
import numpy as np

from control import TimelineControl, pinch_ratio
from calibration import load_calibration, reset_calibration, save_calibration

CALIBRATION_PATH = ROOT / '.local' / 'calibration.json'

WIDTH, HEIGHT = 1280, 800
BG, PANEL = (17, 15, 12), (28, 25, 21)
WHITE, MUTED, ACCENT = (240, 240, 233), (157, 150, 137), (198, 225, 108)
VIDEO_RECT = (24, 130, 900, 506)
CAMERA_RECT = (948, 178, 308, 231)
BAR = (40, 688, 868, 8)
CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7),
               (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
               (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
               (17, 18), (18, 19), (19, 20)]


@dataclass(frozen=True)
class ClipInfo:
    key: str
    name: str
    subtitle: str
    source_label: str
    path: Path


BUILTIN_CLIPS = (
    ClipInfo('1', 'FLOWER', 'A flower. Seven seconds. Your hand.',
             'RED FLOWER / 01:16 - 01:23', ROOT / 'assets/flower.mp4'),
    ClipInfo('2', 'ROCKET', 'A launch. Seven seconds. Your hand.',
             'ROCKET LIFTOFF / 00:18 - 00:25', ROOT / 'assets/rocket.mp4'),
)


def available_clips():
    """Keep local footage when present; fresh clones use generated demo videos."""
    result = []
    for item in BUILTIN_CLIPS:
        if item.path.is_file():
            result.append(item)
        else:
            demo = ROOT / 'assets' / f'demo-{item.name.lower()}.mp4'
            if not demo.is_file():
                raise ValueError('Demo clips missing. Run python setup_assets.py (or setup.command).')
            result.append(replace(item, path=demo,
                                  subtitle='An animation. Seven seconds. Your hand.',
                                  source_label=f'{item.name} / GENERATED DEMO'))
    return result


class Clip:
    """Small JPEG frame cache: arbitrary backward/forward jumps without GOP seeks."""
    def __init__(self, path):
        cap = cv2.VideoCapture(str(path))
        self.frames = []
        self.bytes = 0
        self.current_index = -1
        self.current_frame = None
        try:
            if not cap.isOpened():
                raise ValueError(f'Cannot open video: {path}')
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(self.fps) or self.fps <= 0:
                raise ValueError('Video has no valid frame rate.')
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                if w > 960:
                    frame = cv2.resize(frame, (960, round(h * 960 / w)), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                if not ok:
                    raise ValueError('Could not cache a video frame.')
                self.frames.append(encoded)
                self.bytes += encoded.nbytes
                if len(self.frames) > 900 or self.bytes > 192 * 1024 * 1024:
                    raise ValueError('Clip is too long. Trim to at most 30 seconds / 900 frames.')
        finally:
            cap.release()
        if not self.frames:
            raise ValueError('Video contains no readable frames.')
        self.duration = (len(self.frames) - 1) / self.fps

    def frame_at(self, position):
        index = round(max(0, min(1, position)) * (len(self.frames) - 1))
        if index != self.current_index:
            self.current_frame = cv2.imdecode(self.frames[index], cv2.IMREAD_COLOR)
            self.current_index = index
        return self.current_frame


class ClipDeck:
    """Preload once so switching never blocks the camera or resets calibration."""
    def __init__(self, selected='1', custom=None):
        self.info = {item.key: item for item in available_clips()}
        if custom is not None:
            self.info['3'] = ClipInfo('3', 'CUSTOM', 'Your clip. Your hand.',
                                      'CUSTOM CLIP', Path(custom))
            selected = '3'
        self.clips = {key: Clip(item.path) for key, item in self.info.items()}
        self.selected = selected

    @property
    def current(self):
        return self.clips[self.selected]

    @property
    def current_info(self):
        return self.info[self.selected]

    def select_key(self, keycode):
        key = next((key for key in self.info if keycode == ord(key)), None)
        if key is None:
            return False
        self.selected = key
        return True


def make_detector(model):
    import mediapipe as mp
    if not model.is_file():
        raise ValueError('Hand model missing. Run python setup_assets.py (or setup.command).')
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model), delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect(detector, bgr, timestamp):
    import mediapipe as mp
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
    return result.hand_landmarks[0] if result.hand_landmarks else None


def label(canvas, text, xy, size=0.55, color=WHITE, thickness=1):
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)


def fit(canvas, source, rect):
    x, y, w, h = rect
    sh, sw = source.shape[:2]
    scale = min(w / sw, h / sh)
    nw, nh = round(sw * scale), round(sh * scale)
    ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
    canvas[oy:oy+nh, ox:ox+nw] = cv2.resize(source, (nw, nh), interpolation=cv2.INTER_AREA)


def draw_hand(frame, landmarks):
    if landmarks is None:
        return frame
    h, w = frame.shape[:2]
    points = [(round(p.x * w), round(p.y * h)) for p in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, points[a], points[b], (130, 175, 115), 1, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 3, ACCENT, -1, cv2.LINE_AA)
    cv2.line(frame, points[4], points[8], ACCENT, 3, cv2.LINE_AA)
    for i in (4, 8):
        cv2.circle(frame, points[i], 8, WHITE, 2, cv2.LINE_AA)
    return frame


def render(clip, control, camera=None, status='SHOW YOUR HAND', message='',
           ratio=None, frozen=False, manual=False, clean=False, clip_info=None,
           choices=BUILTIN_CLIPS):
    clip_info = clip_info or BUILTIN_CLIPS[0]
    canvas = np.full((HEIGHT, WIDTH, 3), BG, dtype=np.uint8)
    if clean:
        canvas[:] = 0
        fit(canvas, clip.frame_at(control.position), (0, 0, WIDTH, HEIGHT))
        return canvas
    label(canvas, 'EXPERIMENT 01  /  COMPUTER VISION', (28, 35), 0.46, MUTED)
    label(canvas, 'TIME CONTROL', (24, 88), 1.6, WHITE, 2)
    label(canvas, clip_info.subtitle, (28, 115), 0.53, MUTED)
    for i, choice in enumerate(choices):
        x = 690 + i * 190
        selected = choice.key == clip_info.key
        cv2.rectangle(canvas, (x, 76), (x + 172, 114), ACCENT if selected else PANEL, -1)
        label(canvas, f'{choice.key}  {choice.name}', (x + 16, 101), 0.48,
              BG if selected else MUTED)
    label(canvas, 'LOCAL / CAMERA ONLY', (995, 43), 0.44, ACCENT)
    cv2.rectangle(canvas, (24, 130), (924, 636), (0, 0, 0), -1)
    fit(canvas, clip.frame_at(control.position), VIDEO_RECT)
    cv2.rectangle(canvas, (948, 130), (1256, 742), PANEL, -1)
    label(canvas, 'HAND TRACKING', (968, 161), 0.5, ACCENT)
    if camera is not None:
        fit(canvas, camera, CAMERA_RECT)
    else:
        label(canvas, 'Camera preview', (1019, 285), 0.55, MUTED)
        label(canvas, 'not connected', (1025, 315), 0.48, MUTED)
    label(canvas, status, (966, 439), 0.48, ACCENT if camera is not None else MUTED)
    label(canvas, 'CLOSE', (969, 479), 0.41, MUTED)
    label(canvas, 'OPEN', (1190, 479), 0.41, MUTED)
    cv2.line(canvas, (974, 499), (1226, 499), (70, 67, 58), 3)
    cv2.circle(canvas, (974 + round(252 * control.position), 499), 6, ACCENT, -1, cv2.LINE_AA)
    label(canvas, f'{round(control.position * 100):02d}%', (968, 546), 1.0, WHITE, 2)
    label(canvas, 'TIMELINE POSITION', (1050, 540), 0.34, MUTED)
    label(canvas, 'Spread fingers > forward', (968, 588), 0.43)
    label(canvas, 'Pinch fingers > rewind', (968, 614), 0.43)
    label(canvas, 'No hand > hold frame', (968, 640), 0.43)
    label(canvas, 'C  calibrate close / open', (968, 685), 0.43, ACCENT)
    label(canvas, 'X  forget calibration', (968, 711), 0.43, MUTED)
    label(canvas, clip_info.source_label, (40, 663), 0.43, MUTED)
    label(canvas, f'{control.position * clip.duration:04.1f}s / {clip.duration:04.1f}s', (747, 663), 0.48)
    x, y, w, h = BAR
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (64, 61, 53), -1)
    px = x + round(w * control.position)
    cv2.rectangle(canvas, (x, y), (px, y+h), ACCENT, -1)
    cv2.circle(canvas, (px, y+h//2), 7, WHITE, -1, cv2.LINE_AA)
    hint = message or ('MANUAL: drag the timeline' if manual else 'Pinch your thumb and index finger. Press C to calibrate.')
    label(canvas, hint, (40, 730), 0.45, ACCENT)
    label(canvas, '1  flower   2  rocket     SPACE  hold     M  manual     H  clean view     F  fullscreen     R  camera     ESC  quit', (28, 780), 0.43, MUTED)
    if frozen:
        label(canvas, 'HOLD', (821, 163), 0.5, ACCENT)
    return canvas


def open_camera(index):
    backend = cv2.CAP_AVFOUNDATION if sys.platform == 'darwin' else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, help='Add a custom clip on key 3 and start with it')
    parser.add_argument('--clip', choices=('flower', 'rocket'), default='flower', help='Clip to show at startup')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--no-camera', action='store_true', help='Manual timeline, no camera access')
    parser.add_argument('--snapshot', type=Path, help='Render UI preview without opening camera/window')
    parser.add_argument('--check', action='store_true', help='Validate clip and model without camera/window')
    parser.add_argument('--camera-check', action='store_true', help='Read camera and detect hands for 5 seconds, without saving images')
    args = parser.parse_args()
    deck = ClipDeck(selected='2' if args.clip == 'rocket' else '1', custom=args.video)
    control = TimelineControl()
    model = ROOT / 'models/hand_landmarker.task'
    if args.snapshot:
        control.seek(0.7)
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.snapshot), render(deck.current, control, status='PREVIEW - CAMERA OFF', manual=True,
                                                    clip_info=deck.current_info, choices=tuple(deck.info.values()))):
            raise ValueError('Could not save preview.')
        print(f'Preview: {args.snapshot}')
        return
    if args.check:
        with make_detector(model) as detector:
            for stamp in (0, 33, 66):
                assert detect(detector, np.zeros((480, 640, 3), np.uint8), stamp) is None
        for key, clip in deck.clips.items():
            for position in (0, 0.5, 1, 0.1, 0.9):
                assert clip.frame_at(position) is not None
            print(f'OK: {deck.info[key].name}, {len(clip.frames)} frames, {clip.fps:g} fps, '
                  f'{clip.bytes / 1024**2:.1f} MiB frame cache.')
        print('Model inference OK.')
        return

    detector = None
    cap = None
    try:
        if not args.no_camera:
            detector = make_detector(model)
            cap = open_camera(args.camera)
        if args.camera_check:
            if cap is None or not cap.isOpened():
                raise ValueError('Camera unavailable. Grant camera access to Terminal/Codex in macOS settings.')
            frames = hands = 0
            start = time.monotonic()
            while time.monotonic() - start < 5:
                ok, frame = cap.read()
                if not ok:
                    break
                frames += 1
                hands += detect(detector, cv2.flip(frame, 1), frames * 33) is not None
            if frames == 0:
                raise ValueError('Camera opened but returned no frames.')
            print(f'Camera OK: {frames} frames read; hand detected in {hands} frames. No images saved.')
            return

        calibration_message = load_calibration(control, CALIBRATION_PATH)
        manual = args.no_camera or cap is None or not cap.isOpened()
        frozen = clean = fullscreen = False
        calibration_closed = None
        ratios = deque(maxlen=8)
        message = ('Camera unavailable. Allow access, then press R.'
                   if manual and not args.no_camera else calibration_message)
        last_time = time.monotonic()
        last_stamp = -1
        dragging = False
        window = 'Time Control'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window, WIDTH, HEIGHT)

        def on_mouse(event, x, y, flags, _):
            nonlocal dragging, manual, message, calibration_closed
            if clean:
                return
            if event == cv2.EVENT_LBUTTONDOWN and 76 <= y <= 114:
                for i, choice in enumerate(deck.info.values()):
                    if 690 + i * 190 <= x <= 862 + i * 190:
                        deck.select_key(ord(choice.key))
                        dragging = False
                        return
            if event == cv2.EVENT_LBUTTONDOWN and 660 <= y <= 710 and 28 <= x <= 924:
                dragging = manual = True
                calibration_closed = None
                message = 'MANUAL: press M to return to hand control.'
            if event == cv2.EVENT_LBUTTONUP:
                dragging = False
            if dragging:
                control.seek((x - BAR[0]) / BAR[2])

        cv2.setMouseCallback(window, on_mouse)
        while True:
            now = time.monotonic()
            dt, last_time = now - last_time, now
            landmarks = None
            ratio = None
            camera = None
            if cap is not None and cap.isOpened():
                ok, camera = cap.read()
                if ok:
                    camera = cv2.flip(camera, 1)
                    stamp = max(last_stamp + 1, int(now * 1000))
                    last_stamp = stamp
                    landmarks = detect(detector, camera, stamp)
                    if landmarks is not None:
                        ratio = pinch_ratio(landmarks, camera.shape[1], camera.shape[0])
                    draw_hand(camera, landmarks)
                else:
                    camera = None
                    cap.release()
                    message = 'Camera disconnected. Press R to retry, or M for manual.'
            if ratio is not None:
                ratios.append(ratio)
            else:
                ratios.clear()
            control.update(ratio, dt, frozen=frozen or manual or calibration_closed is not None)
            status = ('MANUAL' if manual else 'HOLD' if frozen else
                      'HAND DETECTED' if ratio is not None else 'NO HAND - FRAME HELD')
            canvas = render(deck.current, control, camera, status, message, ratio, frozen, manual, clean,
                            clip_info=deck.current_info, choices=tuple(deck.info.values()))
            cv2.imshow(window, canvas)
            key = cv2.waitKeyEx(1)
            if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if deck.select_key(key):
                dragging = False
            elif key == ord(' '):
                frozen = not frozen
            elif key == ord('m'):
                manual = not manual
                calibration_closed = None
                message = 'MANUAL: drag the timeline.' if manual else ''
            elif key == ord('h'):
                clean = not clean
            elif key == ord('f'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            elif key == ord('c'):
                if ratio is None or len(ratios) < 4:
                    message = 'Show your hand steadily, then press C.'
                elif calibration_closed is None:
                    calibration_closed = statistics.median(ratios)
                    ratios.clear()
                    message = 'Now spread thumb + index wide. Press C again.'
                else:
                    try:
                        control.calibrate(calibration_closed, statistics.median(ratios))
                        calibration_closed = None
                        manual = False
                        message = save_calibration(control, CALIBRATION_PATH)
                    except ValueError as error:
                        message = str(error)
            elif key == ord('x'):
                calibration_closed = None
                message = reset_calibration(control, CALIBRATION_PATH)
            elif key == ord('r'):
                if cap is not None:
                    cap.release()
                if detector is None:
                    detector = make_detector(model)
                cap = open_camera(args.camera)
                manual = not cap.isOpened()
                message = 'Camera unavailable. Check macOS permissions.' if manual else ''
    finally:
        if cap is not None:
            cap.release()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, cv2.error) as error:
        print(f'Time Control: {error}', file=sys.stderr)
        sys.exit(1)
