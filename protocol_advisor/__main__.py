"""Entry point: ``python -m protocol_advisor`` (or the ``iot-protocol-advisor`` script)."""

from __future__ import annotations

import sys

from protocol_advisor.engine import Engine


def main() -> int:
    try:
        import tkinter  # noqa: F401
    except ImportError:
        print(
            "Tkinter is not available for this Python.\n"
            "  Debian/Ubuntu:  sudo apt install python3-tk\n"
            "  macOS (Homebrew): brew install python-tk\n"
            "  Fedora:         sudo dnf install python3-tkinter",
            file=sys.stderr,
        )
        return 1

    from protocol_advisor.ui import run

    run(Engine())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
