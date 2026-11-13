Run panorama inference from the repository in the `nova3r` environment on a CUDA GPU:

```bash
python scripts/inference/predict_panorama.py archive/overview.png
```

Defaults load `exp_output/hunyuandit_panorama/checkpoints/epoch=09-val_loss=1.4962.ckpt`, use 100 Euler steps and CFG 5, and write to `demo/outputs/panorama_epoch09_meshudf768_padding15/`:

- `panorama.png`: RGB input padded with black at the bottom to 2:1, preserving source pixels. Odd widths receive one extra right-hand column for an exact integer 2:1 canvas. The DINO processor then scales the padded canvas to 1036×518.
- `pager_geometry.npz`: PaGeR pointmap in the panorama OpenCV frame and its validity mask. Added black canvas pixels are excluded.
- `latents.pt`: generated scene latents, visible geometry latents and normalized PaGeR surface.
- `prediction.ply` and `prediction.glb`: reconstructed mesh.
- `run.json`: arguments, normalization and mesh size.

DINOv2-L and PaGeR (`--pager-checkpoint pager`) use Hugging Face weights (downloaded if absent from the local cache). The ShapeVAE checkpoint comes from `experiment.vae_ckpt` in the config. Models for feature extraction and depth are released before loading the scene generator to reduce GPU memory use.

For another panorama, choose a separate output directory:

```bash
python scripts/inference/predict_panorama.py path/to/panorama.jpg --output-dir demo/outputs/my_panorama
```

The mesh uses normalized decoder coordinates, not calibrated metric scene scale. Meshing bounds come from the axis-aligned minimum and maximum of the normalized PaGeR surface points after statistical outlier removal (20 nearest neighbours, mean neighbour distance cutoff at mean + 2 standard deviations). This bounds-only filter leaves the visible-geometry conditioning unchanged. Configure it with `--bounds-outlier-neighbors` and `--bounds-outlier-std-ratio`, or disable it with `--bounds-outlier-neighbors 0`. Removed-point counts are recorded in `run.json`. PaGeR supplies both visible-geometry conditioning and bounding-box points; there is no separate DA3 inference stage. PaGeR itself uses its upstream DA3-based backbone. Use `--pager-scene-mode indoor|outdoor` for the scale head, `--pager-checkpoint` for a local checkpoint directory or Hugging Face model, and `--pager-repo` for the local PaGeR source. Each side is expanded by 15% of that axis's extent (`--bounds-padding 0.15`), giving 30% total expansion. Default meshing uses MeshUDF at resolution 768 along the longest axis, with proportional grid dimensions on the other axes. The unsigned field is decoded in FP32. Use `--mesh-method latent2mesh` for the other extraction pipeline. The script overwrites files in its output directory.
