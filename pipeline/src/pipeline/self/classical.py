#!/usr/bin/env python3
"""
CLI wrapper for classical baselines on self-collected data.

Example:
  python -m pipeline.self.classical --subject 01 --data-dir data/self --runs 01 02 --sweep
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).with_name("mi_riemann_probe.py")
    sys.path.insert(0, str(script_path.parent))
    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()

