from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import app
from control import TimelineControl
from demo_assets import generate_demos
from setup_assets import ensure_model


class AssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        generate_demos(cls.root / 'assets')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_fresh_install_switches_clips_and_preserves_gesture(self):
        specs = tuple(replace(item, path=self.root / 'assets' / item.path.name)
                      for item in app.BUILTIN_CLIPS)
        with mock.patch.object(app, 'ROOT', self.root), mock.patch.object(app, 'BUILTIN_CLIPS', specs):
            deck = app.ClipDeck()
        control = TimelineControl(position=0.62)
        control.calibrate(0.12, 1.6)
        state = (control.closed, control.opened, control.position)
        initial_ids = {key: id(clip) for key, clip in deck.clips.items()}
        for key in ('2', '1', '2'):
            self.assertTrue(deck.select_key(ord(key)))
            self.assertEqual(deck.selected, key)
            self.assertEqual(id(deck.current), initial_ids[key])
            self.assertEqual((control.closed, control.opened, control.position), state)
            self.assertIn('GENERATED DEMO', deck.current_info.source_label)
            self.assertEqual(len(deck.current.frames), 168)
            first = deck.current.frame_at(0).copy()
            last = deck.current.frame_at(1).copy()
            self.assertGreater(float(np.abs(first.astype(float) - last).mean()), 1)
            self.assertIsNotNone(deck.current.frame_at(0.3))
            self.assertEqual(app.render(deck.current, control, clean=True).shape, (800, 1280, 3))
        self.assertFalse(deck.select_key(ord('9')))
        self.assertEqual(deck.selected, '2')

    def test_setup_preserves_existing_demos_and_local_footage(self):
        path = self.root / 'assets' / 'demo-flower.mp4'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        local = self.root / 'assets' / 'flower.mp4'
        local.write_bytes(b'local footage must not be touched')
        try:
            generate_demos(self.root / 'assets')
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            self.assertEqual(local.read_bytes(), b'local footage must not be touched')
        finally:
            local.unlink()

    def test_unexpected_existing_model_is_not_overwritten(self):
        directory = self.root / 'models'
        directory.mkdir(exist_ok=True)
        path = directory / 'hand_landmarker.task'
        path.write_bytes(b'unexpected model')
        with self.assertRaisesRegex(RuntimeError, 'unexpected checksum'):
            ensure_model(directory)
        self.assertEqual(path.read_bytes(), b'unexpected model')


if __name__ == '__main__':
    unittest.main()
