# Portable room editor

Copy this folder and your `test_scenes` data to your PC. No repository checkout
is required. Use Python 3.11 or 3.12 with a desktop display.

```sh
python -m pip install -r requirements.txt
python room_editor.py "C:/data/scannetpp/test_scenes"
```

On macOS/Linux, pass your local path instead. Running without a path opens a
folder picker (requires Tk). You can also pass a single scene folder.

Expected data:

```text
test_scenes/
  <scene>/
    scans/mesh_aligned_0.05.ply
    test_camera_poses.json
```

The mesh can alternatively sit directly inside the scene folder. Coordinates
must be the original ScanNet++ scan coordinates, with Z vertical, matching the
exported camera poses. No training normalization is applied.

1. The top-down view shows sampled mesh vertices, red camera markers and their
   panorama frame IDs. Frame `00002` means image/panorama 2 in that scene.
2. Click vertices around a room, then click the first vertex to close the polygon.
   Disable the Matplotlib toolbar's pan/zoom modes while drawing. Escape restarts
   polygon drawing. Vertex handles allow adjustments after closing.
3. Enter a room name and optionally change Min Z / Max Z. Selection includes the
   whole vertical prism by default. “Hide top 0.3 m” removes roof points from the
   display only; it does not remove anything from exports.
4. Preview in 3D if desired. Red spheres mark cameras. Close that window to resume.
5. Save, draw the next room and repeat. Previous/Next switches scenes. Saved room
   outlines remain visible until you switch scenes. Unsaved selections are lost
   when switching scenes or redrawing the view.

Outputs default to `<input folder>/rooms/<scene>/<room>/`:

- `mesh.ply`: selected triangles with vertex colors, in original scan coordinates.
- `test_camera_poses.json`: matching test camera poses plus the saved polygon and
  height interval. Includes all six cubemap faces for each selected panorama.
- `test_images.txt`: scene/frame pairs for test images whose camera centers are
  inside the room and height interval. Images themselves are not copied.

Use `--output "C:/data/rooms"` for a different output folder. Existing room
folders are never overwritten. Meshes and source poses are not modified.
Overlapping polygons may export the same faces/cameras into multiple rooms.

Selection keeps triangles by their centroid; it does not slice triangles or cap
open boundaries. For clean separation, draw along wall centers or between rooms.
The tool exports geometry/colors, not semantic labels or texture atlases. Camera
labels are shown in the top-down editor; the 3D preview shows camera positions.
Only cameras in the supplied test pose file are displayed. A camera marker does
not indicate its visibility region or guarantee its image shows only that room.

Updated camera exports include all original scanner poses. Each frame has an
`is_test` boolean: red cameras are in the final test list, blue cameras are not.
Room pose exports preserve both groups and their flags; `test_images.txt` lists
only test frames. Older pose files without flags are treated as all-test.
