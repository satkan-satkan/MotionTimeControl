#!/bin/zsh
set -eu
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Сначала запустите setup.command.'
  read -r '?Нажмите Enter, чтобы закрыть окно.'
  exit 1
fi
if ! .venv/bin/python app.py "$@"; then
  echo 'Не удалось запустить Time Control. Сообщение об ошибке выше.'
  read -r '?Нажмите Enter, чтобы закрыть окно.'
  exit 1
fi
