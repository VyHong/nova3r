#!/usr/bin/env python3
"""Compatibility entry point for the SAGE-10K point-cloud lighting baker."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from bake_sage10k_lighting_to_pointcloud import main


if __name__ == "__main__":
    main()
