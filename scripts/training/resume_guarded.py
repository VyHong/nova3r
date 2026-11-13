"""Resume no-Plucker training; retain its standard checkpoint behavior."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'training')]
QUEUE = ROOT / 'exp_output/hunyuandit_panorama_no_plucker/supervised_queue'

if __name__ == '__main__':
    from training.resume_train import main
    selected = ROOT / 'exp_output/hunyuandit_panorama_no_plucker/checkpoints/last.ckpt'
    sys.argv = [sys.argv[0], '--config', str(ROOT / 'checkpoints/hunyuan_dit/config_hunyuandit_panorama_visible_latent_no_plucker.yaml'), '--devices', '1', '--ckpt', str(selected)]
    main()
    marker = QUEUE / 'training_complete.json'
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps({'completed_utc': time.time(), 'resumed_from': str(selected)}) + '\n')
    os.replace(temporary, marker)
