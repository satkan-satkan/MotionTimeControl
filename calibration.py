"""Local-only calibration persistence. No images, paths or camera identifiers."""
import json
import os
from pathlib import Path
import tempfile

from control import TimelineControl


def load_calibration(control: TimelineControl, path: Path) -> str:
    """Apply a valid saved range; never rewrite missing or invalid settings."""
    try:
        with path.open('rb') as source:
            raw = source.read(4097)
    except FileNotFoundError:
        return ''
    except OSError:
        return 'Saved calibration unreadable. Press C to recalibrate.'
    try:
        if len(raw) > 4096:
            raise ValueError('Settings too large')
        data = json.loads(raw)
        if type(data) is not dict or set(data) != {'version', 'closed', 'opened'}:
            raise ValueError('Invalid settings schema')
        if type(data['version']) is not int or data['version'] != 1:
            raise ValueError('Unsupported settings version')
        if any(type(data[key]) not in (int, float) for key in ('closed', 'opened')):
            raise ValueError('Calibration values must be numbers')
        control.calibrate(data['closed'], data['opened'])
    except (ValueError, UnicodeError, OverflowError):
        return 'Saved calibration invalid. Press C to recalibrate.'
    return 'Saved calibration loaded. Your hand controls time.'


def save_calibration(control: TimelineControl, path: Path) -> str:
    """Replace the file atomically only after a complete, valid calibration."""
    validated = TimelineControl()
    validated.calibrate(control.closed, control.opened)
    data = {'version': 1, 'closed': validated.closed, 'opened': validated.opened}
    temporary = None
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # NamedTemporaryFile uses owner-only permissions on POSIX.
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.calibration-', suffix='.tmp', delete=False) as output:
            temporary = Path(output.name)
            json.dump(data, output, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except OSError:
        return 'Calibrated for this session. Could not save locally.'
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return 'Calibration saved locally. Your hand controls time.'


def reset_calibration(control: TimelineControl, path: Path) -> str:
    """Reset the session and forget the saved range, preserving timeline position."""
    defaults = TimelineControl()
    control.closed, control.opened = defaults.closed, defaults.opened
    try:
        path.unlink(missing_ok=True)
    except OSError:
        return 'Defaults active; saved calibration could not be removed.'
    return 'Saved calibration cleared. Pinch fingers, then press C.'
