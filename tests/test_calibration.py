import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from calibration import load_calibration, reset_calibration, save_calibration
from control import TimelineControl


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / '.local' / 'calibration.json'

    def test_first_launch_does_not_create_settings(self):
        control = TimelineControl()
        self.assertEqual(load_calibration(control, self.path), '')
        self.assertEqual((control.closed, control.opened), (0.18, 1.45))
        self.assertFalse(self.path.parent.exists())

    def test_complete_calibration_survives_restart_without_other_state(self):
        original = TimelineControl(position=0.73)
        original.calibrate(0.12, 1.67)
        self.assertIn('saved locally', save_calibration(original, self.path))
        self.assertEqual(json.loads(self.path.read_text()),
                         {'version': 1, 'closed': 0.12, 'opened': 1.67})
        restarted = TimelineControl()
        self.assertIn('loaded', load_calibration(restarted, self.path))
        self.assertEqual((restarted.closed, restarted.opened), (0.12, 1.67))
        self.assertEqual(restarted.position, 0)
        if os.name == 'posix':
            self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_new_completed_calibration_replaces_previous_range(self):
        control = TimelineControl()
        save_calibration(control, self.path)
        control.calibrate(0.22, 1.8)
        save_calibration(control, self.path)
        restarted = TimelineControl()
        load_calibration(restarted, self.path)
        self.assertEqual((restarted.closed, restarted.opened), (0.22, 1.8))

    def test_invalid_saved_settings_do_not_change_control_or_file(self):
        self.path.parent.mkdir()
        cases = [b'broken json', b'[]', b'null', b'\xff', b'x' * 4097]
        for change in [{'version': 2}, {'version': True}, {'closed': True},
                       {'opened': '1.67'}, {'closed': -1}, {'opened': 0.2},
                       {'opened': float('nan')}, {'opened': float('inf')},
                       {'closed': []}, {'opened': 10**400}, {'extra': 'unexpected'}]:
            cases.append(json.dumps({'version': 1, 'closed': 0.12, 'opened': 1.67, **change}).encode())
        for raw in cases:
            with self.subTest(raw=raw[:50]):
                self.path.write_bytes(raw)
                control = TimelineControl(position=0.42)
                self.assertIn('invalid', load_calibration(control, self.path))
                self.assertEqual((control.closed, control.opened, control.position), (0.18, 1.45, 0.42))
                self.assertEqual(self.path.read_bytes(), raw)

    def test_failed_atomic_save_preserves_last_good_file(self):
        control = TimelineControl()
        save_calibration(control, self.path)
        before = self.path.read_bytes()
        control.calibrate(0.12, 1.67)
        with mock.patch('calibration.os.replace', side_effect=PermissionError):
            self.assertIn('Could not save', save_calibration(control, self.path))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])
        self.assertEqual((control.closed, control.opened), (0.12, 1.67))

    def test_unreadable_settings_keep_defaults(self):
        with mock.patch.object(Path, 'open', side_effect=PermissionError):
            self.assertIn('unreadable', load_calibration(TimelineControl(), self.path))

    def test_reset_forgets_range_across_restart(self):
        control = TimelineControl(position=0.4)
        control.calibrate(0.12, 1.67)
        save_calibration(control, self.path)
        self.assertIn('cleared', reset_calibration(control, self.path))
        self.assertEqual((control.closed, control.opened, control.position), (0.18, 1.45, 0.4))
        self.assertFalse(self.path.exists())
        self.assertEqual(load_calibration(TimelineControl(), self.path), '')
        self.assertIn('cleared', reset_calibration(control, self.path))

    def test_reset_reports_when_saved_file_cannot_be_removed(self):
        control = TimelineControl()
        control.calibrate(0.12, 1.67)
        save_calibration(control, self.path)
        with mock.patch.object(Path, 'unlink', side_effect=PermissionError):
            self.assertIn('could not be removed', reset_calibration(control, self.path))
        self.assertTrue(self.path.exists())
        self.assertEqual((control.closed, control.opened), (0.18, 1.45))


if __name__ == '__main__':
    unittest.main()
