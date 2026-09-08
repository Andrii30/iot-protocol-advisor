#!/usr/bin/env bash
# One command to set up and launch the app on macOS or Debian/Ubuntu.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV=".venv"

if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "Need Python 3.10+. Set PYTHON=/path/to/python3 and retry." >&2
  exit 1
fi

if [ ! -d "$VENV" ]; then
  echo "Creating virtualenv…"
  "$PY" -m venv "$VENV"
fi

# Install deps only when requirements.txt is newer than the last install marker.
MARKER="$VENV/.deps-installed"
if [ ! -f "$MARKER" ] || [ requirements.txt -nt "$MARKER" ]; then
  echo "Installing dependencies…"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q -r requirements.txt
  touch "$MARKER"
fi

if ! "$VENV/bin/python" -c 'import tkinter' 2>/dev/null; then
  echo "Tkinter is missing. Install it, then rerun ./run.sh:" >&2
  case "$(uname -s)" in
    Darwin) echo "  brew install python-tk" >&2 ;;
    Linux)  echo "  sudo apt install python3-tk   # Debian/Ubuntu" >&2 ;;
  esac
  exit 1
fi

exec "$VENV/bin/python" -m protocol_advisor
