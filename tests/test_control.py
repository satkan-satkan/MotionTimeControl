import math
from types import SimpleNamespace
import unittest

from control import TimelineControl, pinch_ratio


class ControlTests(unittest.TestCase):
    def test_endpoints_and_reverse(self):
        c = TimelineControl()
        for _ in range(100):
            c.update(3, 1/30)
        self.assertEqual(c.position, 1)
        for _ in range(100):
            c.update(0, 1/30)
        self.assertEqual(c.position, 0)

    def test_lost_hand_and_freeze_hold_exact_frame(self):
        c = TimelineControl(position=0.37)
        self.assertEqual(c.update(None, 10), 0.37)
        self.assertEqual(c.update(2, 10, frozen=True), 0.37)
        self.assertEqual(c.update(math.nan, 1), 0.37)

    def test_frame_rate_independence(self):
        slow, fast = TimelineControl(), TimelineControl()
        for _ in range(10):
            slow.update(1, 1/30)
        for _ in range(20):
            fast.update(1, 1/60)
        self.assertAlmostEqual(slow.position, fast.position, places=7)

    def test_calibration_rejects_reversed_or_tiny_range(self):
        c = TimelineControl()
        for a, b in [(1, 0), (0.2, 0.3), (math.nan, 2)]:
            with self.assertRaises(ValueError):
                c.calibrate(a, b)
        c.calibrate(0.1, 1.1)
        for _ in range(100):
            c.update(0.6, 1/30)
        self.assertAlmostEqual(c.position, 0.5, delta=0.004)

    def test_aspect_and_distance_invariance(self):
        points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
        points[9].y = 0.25
        points[8].x = 0.6875
        self.assertAlmostEqual(pinch_ratio(points, 640, 480), 1)
        for p in points:
            p.x, p.y = 0.5 + (p.x - 0.5)*0.5, 0.5 + (p.y-0.5)*0.5
        self.assertAlmostEqual(pinch_ratio(points, 640, 480), 1)
        self.assertIsNone(pinch_ratio([], 640, 480))


if __name__ == '__main__':
    unittest.main()
