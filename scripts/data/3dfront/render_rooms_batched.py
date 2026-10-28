#!/usr/bin/env python3
"""Render room jobs concurrently, loading each dwelling once for all its rooms."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path(os.environ.get(
    "BLENDERPROC_PYTHON", sys.executable
))
RENDERER = ROOT / "scripts/data/3dfront/render_front3d_scene.py"
GROUP_RENDERER = ROOT / "scripts/data/3dfront/render_scene_group.py"


class ProgressBar:
    """Small thread-safe progress bar without adding a runtime dependency."""

    def __init__(self, total, completed=0):
        self.total = total
        self.completed = completed
        self.initial_completed = completed
        self.session_completed = 0
        self.started = time.perf_counter()
        self.updates = []
        self.lock = threading.Lock()

    @staticmethod
    def _duration(seconds):
        seconds = max(0, round(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes:02d}m"
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"

    def _display(self):
        now = time.perf_counter()
        elapsed = max(now - self.started, 1e-9)
        fraction = self.completed / self.total if self.total else 1.0
        width = 30
        filled = min(width, round(width * fraction))
        remaining = self.total - self.completed

        # Initial resume skips are real overall progress, but counting them as
        # instantaneous throughput makes the ETA wildly optimistic. Estimate
        # only from work completed during this process. Once enough history is
        # available, use a 15-minute moving rate so the ETA follows current
        # throughput instead of the full-run average.
        recent = [(stamp, amount) for stamp, amount in self.updates
                  if stamp >= now - 15 * 60]
        if self.session_completed == 0:
            rate = 0.0
        elif elapsed < 15 * 60 or len(recent) < 2:
            rate = self.session_completed / elapsed
        else:
            recent_elapsed = min(elapsed, 15 * 60)
            rate = sum(amount for _, amount in recent) / recent_elapsed
        eta = self._duration(remaining / rate) if rate else "calculating"
        bar = "#" * filled + "-" * (width - filled)
        end = "\n" if self.completed >= self.total else ""
        print(
            f"\r[{bar}] {self.completed}/{self.total} rooms "
            f"({fraction:6.2%}) | {rate * 60:.2f} rooms/min | ETA {eta}",
            end=end, file=sys.stderr, flush=True,
        )

    def show(self):
        with self.lock:
            self._display()

    def update(self, amount):
        with self.lock:
            self.completed += amount
            self.session_completed += amount
            self.updates.append((time.perf_counter(), amount))
            self._display()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "artifacts/3dfront_gpu_samples/manifest.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/3dfront_room_batches")
    parser.add_argument("--views", type=int, default=10)
    parser.add_argument("--adaptive-views", action="store_true",
                        help="Use 3 views below 8 m², 5 below 16 m², otherwise --views")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--width", type=int, default=1036,
                        help="Equirectangular width; height is width // 2 (default: 1036x518)")
    parser.add_argument("--encode-workers", type=int, default=4,
                        help="PNG encoder threads inside each renderer")
    parser.add_argument("--hdri-root", type=Path,
                        default=Path(os.environ.get("FRONT3D_HDRI_ROOT",
                                                    "/tmp/datasets/3dfront/hdris")))
    parser.add_argument("--glb-texture-size", type=int, default=1024)
    parser.add_argument("--glb-image-format", choices=("AUTO", "JPEG", "WEBP"), default="WEBP")
    parser.add_argument("--glb-image-quality", type=int, default=80)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", choices=("GPU", "CPU"), default="GPU")
    parser.add_argument("--gpu-backend", choices=("OPTIX", "CUDA"), default="OPTIX")
    parser.add_argument("--denoiser", choices=("OPTIX", "NONE"), default="OPTIX")
    parser.add_argument("--gpu-ids", default="0",
                        help=("Comma-separated physical GPU ids. Workers are assigned round-robin, "
                              "so one large GPU may run several workers."))
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0,
                        help="Render at most N selected rooms; 0 means no limit")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-rooms-per-scene", type=int, default=0,
                        help=("Maximum selected rooms per dwelling; 0 (the default) renders "
                              "every manifest room in the persistent scene process."))
    return parser.parse_args()


def complete(folder: Path, room: str, views: int, width: int, samples: int) -> bool:
    required = [folder / "room.glb", folder / "metadata.json", folder / "extrinsics.json"]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    metadata = json.loads((folder / "metadata.json").read_text())
    actual_views = metadata.get("view_count", 0)
    if not isinstance(actual_views, int) or not 1 <= actual_views <= views:
        return False
    required += [folder / f"{room}_panorama_{index:03d}.png" for index in range(actual_views)]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    metadata = json.loads((folder / "metadata.json").read_text())
    return (metadata.get("requested_view_count", actual_views) == views
            and metadata.get("resolution") == [width, width // 2]
            and metadata.get("render_samples") == samples)


def previous_camera_sampling_failure(folder: Path) -> bool:
    """Return whether an earlier attempt conclusively failed to sample cameras."""
    log_path = folder / "render.log"
    if not log_path.is_file():
        return False
    try:
        tail = log_path.read_text(errors="replace")[-12000:]
    except OSError:
        return False
    # Failures from the old furniture-filtered sampler must be retried.
    return ("floor_sampler=projected_triangle_area_v1" in tail
            and "valid camera poses in 10000 attempts" in tail)


def find_hdri(root: Path, name: str) -> Path:
    """Find an HDR/EXR without assuming Poly Haven's old directory layout."""
    candidates = []
    for extension in (".hdr", ".exr"):
        candidates.extend(root.glob(f"**/{name}*{extension}"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find HDRI {name!r} below {root}. Pass --hdri-root or "
            "set FRONT3D_HDRI_ROOT."
        )
    return sorted(candidates, key=lambda path: (len(path.parts), len(path.name), str(path)))[0]


def render_job(index, job, args, gpu_id, execute=True):
    started = time.perf_counter()
    room = job["room"]
    scene_id = Path(job["scene"]).stem
    folder = args.output / scene_id / room
    folder.mkdir(parents=True, exist_ok=True)
    if not args.overwrite and complete(folder, room, args.views, args.width, args.samples):
        return {"scene": scene_id, "room": room, "status": "skipped",
                "output": str(folder), "elapsed_seconds": 0.0}
    source_output = job.get("source_output", job.get("output"))
    source_metadata_path = (Path(source_output) / "metadata.json"
                            if source_output else Path("/__missing_front3d_source_metadata__"))
    source = json.loads(source_metadata_path.read_text()) if source_metadata_path.is_file() else {}
    hdri = source.get("hdri")
    if hdri is not None and not Path(hdri).is_file():
        old_name = Path(hdri).stem
        for suffix in ("_1k", "_2k", "_4k", "_8k", "_16k"):
            if old_name.endswith(suffix):
                old_name = old_name[:-len(suffix)]
                break
        hdri = str(find_hdri(args.hdri_root, old_name))
    if hdri is None:
        hdri_names = ["kiara_3_morning", "canary_wharf", "kiara_7_late-afternoon",
                      "urban_courtyard", "kiara_5_noon", "modern_buildings",
                      "kiara_3_morning", "modern_buildings_night", "moonlit_golf",
                      "rooftop_night"]
        hdri = str(find_hdri(args.hdri_root, hdri_names[index % len(hdri_names)]))
    ceiling_strengths = [0.35, 0.55, 0.8, 1.1, 1.4, 0.65, 1.0, 0.5, 0.8, 1.1]
    lamp_strengths = [4.0, 6.0, 8.0, 10.0, 12.0, 7.0, 9.0, 12.0, 16.0, 20.0]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu_id),
        PYTHONPATH=str(ROOT / "third_party/BlenderProc"),
        FRONT3D_SCENE=job["scene"],
        FRONT3D_ROOM=room,
        FRONT3D_OUTPUT=str(folder),
        FRONT3D_HDRI=hdri,
        FRONT3D_HDRI_ROTATION=str(source.get("hdri_rotation_degrees", index * 37)),
        FRONT3D_HDRI_STRENGTH=str(source.get("hdri_strength", 0.85)),
        FRONT3D_CEILING_EMISSION_STRENGTH=str(source.get("ceiling_emission_strength", ceiling_strengths[index % len(ceiling_strengths)])),
        FRONT3D_LAMP_EMISSION_STRENGTH=str(source.get("lamp_emission_strength", lamp_strengths[index % len(lamp_strengths)])),
        FRONT3D_TIME_OF_DAY=str(source.get("time_of_day", "night" if index % 10 >= 7 else "day")),
        FRONT3D_ENVIRONMENT=str(source.get("environment", "city" if index % 2 else "landscape")),
        FRONT3D_WINDOW_STYLE=["picture", "double_casement", "triple_panel", "grid",
                              "transom", "awning", "arched_transom"][index % 7],
        FRONT3D_DOOR_STYLE=["flush", "two_panel", "six_panel", "glazed", "louvered"][index % 5],
        FRONT3D_VIEWS=str(args.views),
        FRONT3D_ADAPTIVE_VIEWS="1" if args.adaptive_views else "0",
        FRONT3D_SAMPLES=str(args.samples),
        FRONT3D_WIDTH=str(args.width),
        FRONT3D_ENCODE_WORKERS=str(args.encode_workers),
        FRONT3D_CAMERA_SEED=str(42 + index),
        FRONT3D_CAMERA_SAMPLER="legacy",
        FRONT3D_RENDER_DEVICE=args.device,
        FRONT3D_GPU_BACKEND=args.gpu_backend,
        FRONT3D_DENOISER=args.denoiser,
        FRONT3D_GLB_TEXTURE_MAX_SIZE=str(args.glb_texture_size),
        FRONT3D_GLB_IMAGE_FORMAT=args.glb_image_format,
        FRONT3D_GLB_IMAGE_QUALITY=str(args.glb_image_quality),
    )
    if not execute:
        return {"scene": scene_id, "room": room, "output": str(folder),
                "gpu": gpu_id, "env": env}
    with (folder / "render.log").open("w") as log:
        result = subprocess.run(
            [str(PYTHON), str(RENDERER)], cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT,
        )
    if not complete(folder, room, args.views, args.width, args.samples):
        tail = (folder / "render.log").read_text()[-4000:]
        raise RuntimeError(f"{room} failed with exit {result.returncode}:\n{tail}")
    return {"scene": scene_id, "room": room, "status": "rendered",
            "output": str(folder), "gpu": gpu_id,
            "elapsed_seconds": time.perf_counter() - started}


def render_scene_group(scene_jobs, args, gpu_id):
    """Run every incomplete room from one dwelling in one Blender process."""
    results = []
    pending = []
    for index, job in scene_jobs:
        room = job["room"]
        scene_id = Path(job["scene"]).stem
        folder = args.output / scene_id / room
        if not args.overwrite and complete(folder, room, args.views, args.width, args.samples):
            results.append({"scene": scene_id, "room": room, "status": "skipped",
                            "output": str(folder), "elapsed_seconds": 0.0})
        elif not args.overwrite and previous_camera_sampling_failure(folder):
            results.append({"scene": scene_id, "room": room,
                            "status": "skipped_camera_sampling",
                            "output": str(folder), "elapsed_seconds": 0.0})
        else:
            pending.append(render_job(index, job, args, gpu_id, execute=False))
    if not pending:
        return results

    started = time.perf_counter()
    process = subprocess.run(
        [str(PYTHON), str(GROUP_RENDERER)], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id),
                 PYTHONPATH=str(ROOT / "third_party/BlenderProc")),
        input=json.dumps(pending), text=True, capture_output=True,
    )
    elapsed = time.perf_counter() - started
    incomplete = [
        item for item in pending
        if not complete(Path(item["output"]), item["room"], args.views,
                        args.width, args.samples)
    ]
    if process.returncode and incomplete:
        raise RuntimeError(
            f"Persistent renderer failed with exit {process.returncode} for "
            f"{Path(scene_jobs[0][1]['scene']).stem}:\n"
            f"{process.stdout[-2000:]}\n{process.stderr[-4000:]}"
        )
    per_room_elapsed = elapsed / len(pending)
    for item in pending:
        folder = Path(item["output"])
        if not complete(folder, item["room"], args.views, args.width, args.samples):
            tail = (folder / "render.log").read_text()[-4000:]
            raise RuntimeError(f"{item['room']} did not produce complete outputs:\n{tail}")
        results.append({
            "scene": item["scene"], "room": item["room"], "status": "rendered",
            "output": item["output"], "gpu": gpu_id,
            "elapsed_seconds": per_room_elapsed,
            "scene_group_elapsed_seconds": elapsed,
            "scene_group_room_count": len(pending),
        })
    return results


def main():
    args = parse_args()
    if min(args.views, args.samples, args.width, args.workers, args.shard_count,
           args.encode_workers) < 1:
        raise SystemExit("views, samples, width, workers, and shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard-index must be in [0, shard-count)")
    gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        raise SystemExit("at least one GPU id is required")
    if args.max_rooms_per_scene < 0 or args.limit < 0:
        raise SystemExit("max-rooms-per-scene and limit must be non-negative")
    jobs = json.loads(args.manifest.read_text())
    jobs = [(index, job) for index, job in enumerate(jobs)
            if index % args.shard_count == args.shard_index]
    if args.limit:
        jobs = jobs[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for index, job in jobs:
        grouped[str(Path(job["scene"]).resolve())].append((index, job))
    if args.max_rooms_per_scene > 0:
        grouped = {
            scene: scene_jobs[:args.max_rooms_per_scene]
            for scene, scene_jobs in grouped.items()
        }
    scene_groups = sorted(grouped.values(), key=lambda group: group[0][0])
    total_rooms = sum(len(group) for group in scene_groups)

    # Credit resumable outputs before launching workers. This makes the bar
    # start at the true filesystem state and keeps skipped rooms out of the ETA
    # throughput calculation.
    initial_results = []
    pending_groups = []
    for group in scene_groups:
        pending = []
        for index, job in group:
            room = job["room"]
            scene_id = Path(job["scene"]).stem
            folder = args.output / scene_id / room
            if not args.overwrite and complete(
                    folder, room, args.views, args.width, args.samples):
                initial_results.append({
                    "scene": scene_id, "room": room, "status": "skipped",
                    "output": str(folder), "elapsed_seconds": 0.0,
                })
            elif not args.overwrite and previous_camera_sampling_failure(folder):
                initial_results.append({
                    "scene": scene_id, "room": room,
                    "status": "skipped_camera_sampling",
                    "output": str(folder), "elapsed_seconds": 0.0,
                })
            else:
                pending.append((index, job))
        if pending:
            pending_groups.append(pending)

    progress = ProgressBar(total_rooms, completed=len(initial_results))
    print(f"Resume scan: {len(initial_results)} already terminal, "
          f"{total_rooms - len(initial_results)} remaining", file=sys.stderr)
    progress.show()

    # Each worker has a serial scene queue. Workers are mapped round-robin to
    # devices, allowing high-memory GPUs to run multiple Cycles processes.
    assignments = [[] for _ in range(args.workers)]
    for group_index, group in enumerate(pending_groups):
        assignments[group_index % args.workers].append(group)

    def run_gpu_queue(worker_index):
        queue_results = []
        gpu_id = gpu_ids[worker_index % len(gpu_ids)]
        for group in assignments[worker_index]:
            try:
                group_results = render_scene_group(group, args, gpu_id)
            except Exception as error:
                group_results = []
                for _, job in group:
                    room = job["room"]
                    scene_id = Path(job["scene"]).stem
                    folder = args.output / scene_id / room
                    if complete(folder, room, args.views, args.width, args.samples):
                        status = "rendered"
                        item = {"scene": scene_id, "room": room, "status": status,
                                "output": str(folder), "gpu": gpu_id}
                    else:
                        item = {"scene": scene_id, "room": room, "status": "failed",
                                "output": str(folder), "gpu": gpu_id,
                                "error": str(error)}
                    group_results.append(item)
            queue_results.extend(group_results)
            progress.update(len(group_results))
        return queue_results

    results = initial_results
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_gpu_queue, worker): worker
                   for worker in range(args.workers)}
        for future in as_completed(futures):
            for result in future.result():
                results.append(result)
                print(json.dumps(result), flush=True)
    results.sort(key=lambda item: (item.get("scene", ""), item["room"]))
    (args.output / "batch_results.json").write_text(json.dumps(results, indent=2) + "\n")
    samples = []
    for result in results:
        folder = Path(result["output"])
        if result.get("status") == "failed" or not (folder / "metadata.json").is_file():
            continue
        metadata = json.loads((folder / "metadata.json").read_text())
        relative = folder.relative_to(args.output)
        samples.extend((relative / name).as_posix() for name in metadata["panoramas"])
    (args.output / "panorama_list.json").write_text(json.dumps(samples, indent=2) + "\n")


if __name__ == "__main__":
    main()
