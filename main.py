"""Top-level CLI entrypoint. See spec §3.3."""

from __future__ import annotations

import sys

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())
