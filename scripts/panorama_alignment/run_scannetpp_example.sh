#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python scripts/panorama_alignment/align_panorama.py \
  --mesh exp_output/hunyuandit_panorama/eval_last_ckpt_mesh_udf_1024_09c1414f1b_00003_padding10_seed43/padding_0.1/nova3r_training/version_0/test/meshes/scannetpp_09c1414f1b_00003_4/test_mesh.ply \
  --png datasets/scannetpp/panos/updated_final_test_100_prepared_erp/09c1414f1b/panocam/images/3.png \
  "$@"
