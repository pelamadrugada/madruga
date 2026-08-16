#!/usr/bin/env bash
# Instala o que falta na primeira vez e roda o nitido.
#
#   ./rodar.sh foto.jpg                    -> grava foto_5x.jpg ao lado
#   ./rodar.sh foto.jpg -o grande.png
#   ./rodar.sh pasta/ -o saida/
#   ./rodar.sh clipe.mp4 -o clipe_5x.mp4
#
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "primeira execucao: preparando o ambiente..."
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
fi

exec .venv/bin/python -m nitido "$@"
