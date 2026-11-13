# Pick an exact mesh viewpoint in Blender

Requires Blender 4.x or 5.x. The script uses Blender's built-in Python modules;
no package installation is required. Run it on the computer where you want to
navigate the mesh. Copy the script and mesh there if needed.

1. Open Blender and switch to **Scripting**.
2. In the Text Editor, choose **Open** and select `mesh_view_picker.py`.
3. Click **Run Script** (Alt-P while the pointer is in the Text Editor).
4. Select your mesh in the file dialog. For these NOVA3R PLY files, leave
   **Source is Y-up** enabled. It opens in a new scene, preserving existing scenes.
5. Return to **Layout**. Hover over the 3D viewport, press **N**, and select
   the **Mesh View** sidebar tab.
6. Navigate to your desired position. Middle mouse orbits, Shift-middle mouse
   pans, and the wheel zooms. **Shift-`** starts Walk Navigation: use WASD,
   Q/E for down/up, move the mouse to look, and click to confirm.
7. Set the output **Width** and **Height**, then click **Capture Current View**.
   This displays the camera frame and enables **Navigate camera**. Continue
   navigating to refine the composition. Only what is inside that frame will
   appear in the output. You can change the lens or select an orthographic view.
8. Click **Export View JSON**, choose a filename, and save. The same JSON is
   copied to the clipboard. Share that file or paste the text with the mesh path.

The panel displays camera position and rotation. The JSON additionally records
the full camera transform, focal length, sensor dimensions, projection, lens
shift, clipping distances, output dimensions, pixel aspect, and mesh transforms.
These settings reproduce camera geometry and framing; material, lighting,
ceiling cuts, and color management remain separate rendering choices.

The original mesh file is never modified. The preview uses uniform grey;
there is no remeshing, decimation, or geometry smoothing.

## Replay a saved view

With a Blender executable installed:

```bash
blender --background --python scripts/visualization/mesh_view_picker.py -- \
  --view /path/to/view.json --mesh /path/to/test_mesh.ply \
  --render /path/to/preview.png
```

`--mesh` is optional; use it when the mesh path differs between machines.
This produces a quick Workbench preview from the exported camera. For a final
Cycles render, the same camera and mesh transforms can be restored with the
script's `restore_view` function and combined with the desired lighting.
