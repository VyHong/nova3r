#!/usr/bin/env python3
"""Run multiple rooms from one 3D-FRONT dwelling in one Blender process."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import runpy
import sys
import traceback


ROOT = Path(__file__).resolve().parents[3]
RENDERER = ROOT / "scripts/data/3dfront/render_front3d_scene.py"


def main() -> None:
    # BlenderProc's external-bpy cleanup handler consults this at interpreter
    # shutdown, so it must outlive each individual room environment.
    os.environ.setdefault("USE_EXTERNAL_BPY_MODULE", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/blenderproc-matplotlib")
    jobs = json.load(sys.stdin)
    if not jobs:
        return
    scenes = {str(Path(job["env"]["FRONT3D_SCENE"]).resolve()) for job in jobs}
    if len(scenes) != 1:
        raise ValueError("A persistent render group must contain exactly one dwelling")

    failures = []
    for job in jobs:
        output = Path(job["env"]["FRONT3D_OUTPUT"])
        output.mkdir(parents=True, exist_ok=True)
        old_environment = os.environ.copy()
        os.environ.update(job["env"])
        os.environ["FRONT3D_PERSISTENT_WORKER"] = "1"
        try:
            with (output / "render.log").open("w") as log:
                with redirect_stdout(log), redirect_stderr(log):
                    runpy.run_path(str(RENDERER), run_name="__main__")
        except BaseException as error:  # preserve the other rooms' diagnostics
            failures.append({"room": job["room"], "error": repr(error)})
            with (output / "render.log").open("a") as log:
                traceback.print_exc(file=log)
            break
        finally:
            os.environ.clear()
            os.environ.update(old_environment)

    if failures:
        print(json.dumps(failures, indent=2), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
