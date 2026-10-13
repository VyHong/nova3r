# Evaluation

Evaluation code is grouped by purpose:

- `evaluate_pcd.py`: evaluate a predicted point cloud against ground truth.
- `evaluate_panorama_metrics.py`: compute panorama-level image metrics.
- `mv_recon/`: multi-view reconstruction metrics and the SCRREAM evaluator.
- `baselines/`: unified evaluation runners and adapters for external baselines.
- `notebooks/`: interactive result analysis and visualization.

## Point-cloud evaluation

```bash
python eval/evaluate_pcd.py \
  --gt_ply /path/to/ground_truth.ply \
  --pred_ply /path/to/prediction.ply \
  --output_file /path/to/results.json
```

## Multi-view reconstruction evaluation

Use the maintained launchers from the repository root:

```bash
bash scripts/eval/eval_scrream_n1_stage2.sh --data_root /path/to/datasets
bash scripts/eval/eval_scrream_n2_stage2.sh --data_root /path/to/datasets
```

## Baselines

See [`baselines/README.md`](baselines/README.md) for setup and invocation.
