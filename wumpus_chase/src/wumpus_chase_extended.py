#!/usr/bin/env python3
"""Entry point for extended-only training (delegates to unified CLI)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cli.train import main


if __name__ == "__main__":
    main()
