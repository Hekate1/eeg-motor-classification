#!/usr/bin/env python3
"""
CLI wrapper for training on self-collected data.

Example:
  python -m pipeline.self.train --data-dir data/self --runs 01 02 --model-name scratch_demo
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).with_name("train_offline_model.py")

    # Ensure local helper imports resolve if present.
    sys.path.insert(0, str(script_path.parent))

    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()

