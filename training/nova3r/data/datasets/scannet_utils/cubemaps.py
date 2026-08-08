"""Compatibility entry point for tools/cubemap/scannetpp.py."""

from pathlib import Path
import runpy
import sys


target = Path(__file__).resolve().parents[5] / "tools" / "cubemap" / "scannetpp.py"
sys.path.insert(0, str(target.parent))
runpy.run_path(str(target), run_name="__main__")
