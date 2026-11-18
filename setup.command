#!/bin/zsh
set -eu
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  if command -v python3.12 >/dev/null 2>&1; then
    python3.12 -m venv .venv
  else
    python3 -m venv .venv
  fi
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 for the pinned dependencies."'
.venv/bin/python -m pip install --disable-pip-version-check --timeout 120 -r requirements.txt
.venv/bin/python setup_assets.py
echo 'Готово. Запустите run.command.'
