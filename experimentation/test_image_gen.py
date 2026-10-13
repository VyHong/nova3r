"""Compatibility entry point for scripts/data/cubemap/legacy_views.py."""

from pathlib import Path
import runpy
import sys


target = Path(__file__).resolve().parents[1] / "scripts" / "data" / "cubemap" / "legacy_views.py"
sys.path.insert(0, str(target.parent))
runpy.run_path(str(target), run_name="__main__")
