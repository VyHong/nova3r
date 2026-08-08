#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${1:-/tmp/dataset/sage10k}"
OUTPUT_ZIP="${2:-/mnt/projects/theses/vyhong/sage-10k/panorama-scenes.zip}"
THREADS="${THREADS:-$(nproc)}"

if [[ ! "$THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: THREADS must be a positive integer." >&2
    exit 1
fi

command -v 7z >/dev/null || {
    echo "Error: 7z is required." >&2
    exit 1
}

command -v python >/dev/null || {
    echo "Error: python is required." >&2
    exit 1
}

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: source directory does not exist: $SOURCE_DIR" >&2
    exit 1
fi

SOURCE_DIR="$(realpath "$SOURCE_DIR")"
mkdir -p "$(dirname "$OUTPUT_ZIP")"
OUTPUT_ZIP="$(realpath -m "$OUTPUT_ZIP")"
PARTIAL_ZIP="${OUTPUT_ZIP}.partial"

if [[ -e "$OUTPUT_ZIP" || -e "$PARTIAL_ZIP" ]]; then
    echo "Error: output or partial output already exists:" >&2
    echo "  $OUTPUT_ZIP" >&2
    echo "  $PARTIAL_ZIP" >&2
    exit 1
fi

STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sage10k-unified.XXXXXXXX")"

cleanup() {
    find "$STAGE_DIR" -mindepth 1 -delete 2>/dev/null || true
    rmdir "$STAGE_DIR" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Source:  $SOURCE_DIR"
echo "Output:  $OUTPUT_ZIP"
echo "Threads: $THREADS"
echo "Checking scenes and creating a hard-link staging tree..."

python - "$SOURCE_DIR" "$STAGE_DIR/sage10k" <<'PY'
import json
import os
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
target.mkdir(parents=True)

complete = 0
incomplete = 0
duplicates = 0

for marker in sorted(source.rglob(".complete.json")):
    scene = marker.parent
    name = scene.name

    try:
        metadata = json.loads(marker.read_text())
        count = metadata.get("num_panoramas")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("invalid num_panoramas")

        expected = {
            f"{name}_panorama_{index:03d}.png" for index in range(count)
        }
        extrinsics_path = scene / "extrinsics.json"
        extrinsics = json.loads(extrinsics_path.read_text())
        scene_glb = scene / "scene.glb"

        valid = (
            set(extrinsics) == expected
            and scene_glb.is_file()
            and scene_glb.stat().st_size > 0
            and all(
                (scene / filename).is_file()
                and (scene / filename).stat().st_size > 0
                for filename in expected
            )
        )
    except (OSError, ValueError, TypeError):
        valid = False

    if not valid:
        incomplete += 1
        continue

    destination = target / name
    if destination.exists():
        duplicates += 1
        continue

    try:
        destination.mkdir()
        files = [marker, extrinsics_path, scene_glb]
        files.extend(scene / filename for filename in sorted(expected))
        for path in files:
            os.link(path, destination / path.name)
    except OSError as error:
        raise SystemExit(
            f"Could not hard-link {scene} into the staging tree: {error}"
        ) from error
    complete += 1

print(f"Complete scenes:                 {complete}")
print(f"Invalid/incomplete marked scenes: {incomplete}")
print(f"Duplicate complete scene names:   {duplicates}")

if complete == 0:
    raise SystemExit("No complete scenes were found")
PY

echo "Creating ZIP with fast Deflate compression..."
(
    cd "$STAGE_DIR"
    7z a -tzip -mx=1 -mmt="$THREADS" "$PARTIAL_ZIP" sage10k
)

echo "Verifying ZIP CRCs..."
7z t -tzip -mmt="$THREADS" "$PARTIAL_ZIP"

mv -- "$PARTIAL_ZIP" "$OUTPUT_ZIP"
echo "Created and verified: $OUTPUT_ZIP"
