# Manual panorama alignment

Run on a machine with a desktop display, using a Python environment with
`numpy scipy pillow trimesh matplotlib open3d` and an interactive Matplotlib
backend (for example Qt with `PySide6`, or Tk).

```bash
python scripts/panorama_alignment/align_panorama.py \
  --mesh /path/to/test_mesh.ply \
  --png /path/to/panorama.png
```

For the current ScanNet++ example, use the aligned `test_mesh.ply` and
`datasets/scannetpp/panos/updated_final_test_100_prepared_erp/09c1414f1b/panocam/images/3.png`.
The script is standalone and can be copied to your desktop machine.
A remote terminal requires a working graphical display/X forwarding.

1. The preview renders solid triangles from the panorama camera position. Drag either pane to look around; use the mouse wheel to change the perspective field of view.
2. Start with **Yaw Y**, then adjust pitch/roll. Move Camera X/Y/Z to align
   object boundaries. Click a slider and use arrow keys for small adjustments.
   Increase `--move-range 5` if the position range is too small.
3. Toggle **Geometry only** to inspect the shaded structure without panorama colors. This affects only the preview.
4. **Save pose** writes `panorama_pose.json`.
5. **Export colored mesh** writes `mesh_panorama_aligned.ply`, `.glb`, and
   `visibility.npz`. Export can take a minute for large meshes and temporarily
   blocks the window. Progress appears in the terminal.

Default output directory: `<mesh directory>/manual_panorama_alignment/`.
Use `--output /path/to/folder` to change it, and
`--pose /path/to/panorama_pose.json` to reload saved settings.
The source mesh and panorama are never overwritten.

Meshes must use Hunyuan camera axes (X right, Y up, Z backward), in metres.
The panorama must represent a full 360 by 180 degree ERP. For cropped ScanNet++
images, use the prepared 2:1 PNG with bottom padding. A nearby `metrics.json`
with `mesh_alignment_applied: true` initializes camera position from its
`alignment_shift`; otherwise position and rotation start at zero.

The interactive preview raycasts the full mesh into a perspective image and uses
headlight shading to reveal structure. The left pane shows the panorama in the
same viewing direction, and the right pane shows shaded mesh geometry. Viewing
rotation is separate from the panorama pose: dragging looks around without
changing saved alignment. The camera position sliders move the viewing origin.
Use `--preview-width 640 --preview-height 480` for more detail or smaller values
for faster updates. Mesh acceleration is built once when opening the editor.
Export uses the existing 2 mm visibility / 8 cm local proximity / 10 cm depth gap /
0.005 m² small island / 4 cm island proximity settings and leaves hidden vertices
slate blue. Changing camera rotation affects image lookup;
changing its position also changes visibility. Rotation is extrinsic XYZ Euler
in degrees, mapping the Hunyuan panorama camera into the mesh frame.
A single camera pose cannot correct differences in individual reconstructed objects.

## Landmark fitting

1. **Shift + left-click** a clear corner in the panorama pane (left).
2. **Shift + left-click** its matching surface in the mesh pane (right).
   Enable **Geometry only** if texture makes the mesh corner difficult to identify.
   Numbered markers show completed pairs; a red `+` indicates a pending panorama pick.
3. Repeat for **8–12 pairs**, spread around the room at different heights and
   distances. Ordinary dragging turns the view without picking, including between
   the two clicks of a pair. Avoid the black padded part of the panorama.
4. Click **Fit camera**. At least six pairs are required. The fit optimizes camera
   position and rotation using robust spherical direction residuals, keeping mesh
   geometry fixed. Pose sliders and previews update immediately. Pose and landmark
   files are saved automatically after fitting.
5. Inspect the median/max angular errors in the status line. Individual errors,
   in pair order, are stored as `angular_errors_degrees` in `landmarks.json`.
   Large errors may indicate wrong correspondences or inaccurate reconstructed
   geometry. **Undo landmark** cancels the pending pick or removes the last pair.
6. Export the colored mesh when satisfied.

Pairs are automatically saved after completing or undoing a pair. **Save
landmarks** also saves them explicitly. Reload a session using both files:

```bash
python scripts/panorama_alignment/align_panorama.py \
  --mesh /path/to/test_mesh.ply --png /path/to/panorama.png \
  --pose /path/to/panorama_pose.json \
  --landmarks /path/to/landmarks.json
```

Landmarks store fixed 3D mesh points and panorama-frame unit directions, so they
survive pose adjustments and looking around. Reloading checks that input paths
match. Fitting uses the current sliders as its initial estimate; roughly align
large orientation offsets manually first. No landmark or camera fit can correct
all differences in independently reconstructed object shapes.
