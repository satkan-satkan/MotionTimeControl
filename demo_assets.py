"""Original procedural demo animations. No external footage or camera input."""
from pathlib import Path
import math

import cv2
import numpy as np

SIZE = (640, 360)
FPS, SECONDS = 24, 7


def flower_frame(progress):
    frame = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    cv2.line(frame, (320, 340), (320, 173), (82, 137, 39), 7, cv2.LINE_AA)
    cv2.ellipse(frame, (285, 270), (42, 12), 28, 0, 360, (56, 115, 35), -1, cv2.LINE_AA)
    cv2.ellipse(frame, (354, 293), (38, 11), -28, 0, 360, (56, 115, 35), -1, cv2.LINE_AA)
    opening = progress * progress * (3 - 2 * progress)
    for i in range(12):
        angle = 2 * math.pi * i / 12
        radius = 8 + 53 * opening
        center = (round(320 + math.cos(angle) * radius), round(157 + math.sin(angle) * radius))
        axes = (round(13 + 36 * opening), round(36 - 15 * opening))
        color = (90 + i * 6, 45 + i * 2, 225)
        cv2.ellipse(frame, center, axes, math.degrees(angle), 0, 360, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (320, 157), round(4 + 19 * opening), (70, 200, 255), -1, cv2.LINE_AA)
    return frame


def rocket_frame(progress):
    frame = np.empty((SIZE[1], SIZE[0], 3), np.uint8)
    for y in range(SIZE[1]):
        frame[y, :] = (48 + y // 7, 27 + y // 9, 17 + y // 14)
    cv2.rectangle(frame, (0, 326), (640, 360), (35, 39, 34), -1)
    cv2.rectangle(frame, (270, 317), (370, 326), (105, 109, 100), -1)
    # Smoke and flight are deterministic functions of the timeline, so rewind is exact.
    for i in range(12):
        spread = (15 + progress * 150) * math.sin(i * 2.4)
        center = (round(320 + spread), round(320 - progress * 24 * abs(math.cos(i))))
        cv2.circle(frame, center, round(3 + progress * (15 + i)), (124, 131, 129), -1, cv2.LINE_AA)
    base = round(312 - 225 * progress ** 1.4)
    flame = np.array([(308, base), (320, base + round(12 + 27 * progress)), (332, base)], np.int32)
    cv2.fillConvexPoly(frame, flame, (35, 143, 255), cv2.LINE_AA)
    cv2.rectangle(frame, (307, base - 63), (333, base), (223, 231, 233), -1)
    cv2.fillConvexPoly(frame, np.array([(307, base - 63), (320, base - 88), (333, base - 63)]), (100, 165, 235), cv2.LINE_AA)
    cv2.fillConvexPoly(frame, np.array([(307, base - 22), (295, base + 3), (307, base - 2)]), (95, 146, 215), cv2.LINE_AA)
    cv2.fillConvexPoly(frame, np.array([(333, base - 22), (345, base + 3), (333, base - 2)]), (95, 146, 215), cv2.LINE_AA)
    cv2.circle(frame, (320, base - 43), 8, (220, 140, 60), -1, cv2.LINE_AA)
    return frame


def generate_demos(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, renderer in [('flower', flower_frame), ('rocket', rocket_frame)]:
        target = directory / f'demo-{name}.mp4'
        if target.exists():
            continue
        temp = target.with_name(f'demo-{name}.tmp.mp4')
        writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*'mp4v'), FPS, SIZE)
        try:
            if not writer.isOpened():
                raise RuntimeError('OpenCV cannot write MP4. Install opencv-contrib-python with FFmpeg support.')
            for index in range(FPS * SECONDS):
                frame = renderer(index / (FPS * SECONDS - 1))
                cv2.putText(frame, 'GENERATED DEMO', (18, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (170, 180, 180), 1, cv2.LINE_AA)
                writer.write(frame)
        finally:
            writer.release()
        # Validate output before promoting a partially written file to a demo asset.
        capture = cv2.VideoCapture(str(temp))
        try:
            count = 0
            while capture.read()[0]:
                count += 1
            if count != FPS * SECONDS:
                raise RuntimeError(f'Demo encoding failed: expected 168 frames, got {count}.')
        finally:
            capture.release()
        temp.replace(target)
        print(f'Created {target.name}')
