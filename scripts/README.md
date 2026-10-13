# Scripts

This directory contains executable project workflows:

- `data/`: dataset download, preparation, and maintenance utilities.
- `eval/`: evaluation launchers.
- `inference/`: the Nova3R demo and standalone prediction programs.
- `preprocessing/`: feature, latent, and geometry preparation programs.
- `training/`: helpers launched by training workflows.
- `visualization/`: interactive inspection and visualization programs.

The root-level shell launchers remain at their existing paths for compatibility:
`setup.sh`, `podman.sh`, `train.sh`, `train_ddp.sh`, `test.sh`,
`bulk_mount_mount_zip.sh`, and `mount_datasets.sh`.

Reusable Python modules stay under `nova3r/` or `training/`; only standalone
command-line programs belong here.

## Layout

- `data/3dfront/`: 3D-FRONT panorama and anonymization rendering.
- `data/cubemap/`: cubemap generation and conversion.
- `data/sage10k/`: SAGE-10K maintenance and rendering.
- `preprocessing/features/`: image-feature extraction.
- `preprocessing/latents/`: ShapeVAE latent encoding and decoding.
- `preprocessing/mesh2sdf/`: surface, UDF, and watertight-mesh preparation.
