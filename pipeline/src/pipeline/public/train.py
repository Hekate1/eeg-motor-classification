#!/usr/bin/env python3
"""
CLI wrapper for `pipeline.public.run` (script-style entrypoint).

Example:
  python -m pipeline.public.train --pretrain
"""

from __future__ import annotations

import runpy
import sys
import os
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).with_name("run.py")

    os.environ.setdefault("PUBLIC_PROCESSED_DATA_DIR", "data/public/processed_data")

    # Ensure script-style intra-directory imports like `import data_utils` work.
    sys.path.insert(0, str(script_path.parent))

    runpy.run_path(str(script_path), run_name="__main__")


if __name__ == "__main__":
    main()

