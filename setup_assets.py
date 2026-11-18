"""Prepare local demo clips and download the verified official hand model."""
import argparse
import hashlib
from pathlib import Path
import shutil
import ssl
import urllib.request

ROOT = Path(__file__).resolve().parent
MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
MODEL_SHA256 = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(directory):
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / 'hand_landmarker.task'
    if destination.exists():
        if sha256(destination) != MODEL_SHA256:
            raise RuntimeError('Existing hand model has an unexpected checksum; move it aside before retrying.')
        print('Hand model verified.')
        return
    import certifi
    context = ssl.create_default_context(cafile=certifi.where())
    temporary = destination.with_suffix('.task.part')
    try:
        with urllib.request.urlopen(MODEL_URL, context=context, timeout=120) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output)
        if sha256(temporary) != MODEL_SHA256:
            raise RuntimeError('Downloaded hand model checksum mismatch; no model was installed.')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print('Official hand model downloaded and verified.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo-only', action='store_true', help='Generate demo clips offline; skip model download')
    args = parser.parse_args()
    from demo_assets import generate_demos
    generate_demos(ROOT / 'assets')
    if not args.demo_only:
        ensure_model(ROOT / 'models')
    print('Ready. Run python app.py (or python app.py --no-camera).')


if __name__ == '__main__':
    main()
