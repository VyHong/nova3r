# Mesh2SDF support files

User-facing Python programs were moved to `scripts/preprocessing/mesh2sdf/` and
dataset-specific programs to `scripts/data/`.

This directory intentionally retains only:

- `mesh_udf/`, which is imported by the training runtime and includes a compiled
  marching-cubes extension;
- existing shell wrappers under `3dfront/` and `sage-10k/`, whose paths remain
  stable for compatibility;
- local dependency archives used by the SAGE-10K environment.
