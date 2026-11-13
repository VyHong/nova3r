"""Finish LaRI all-layer per-face Chamfer evaluations, retaining completed results."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'eval_results/baselines/lari'
SUITE = BASE / 'merged_layers_per_face_suite'


def counts(dataset):
    path = BASE / f'{dataset}_independent_faces_cube90_merged_layers_per_face/results.json'
    try:
        rows = json.loads(path.read_text())
    except FileNotFoundError:
        return 0, 0
    return len({r['sample'] for r in rows if r.get('status') == 'ok'}), sum(r.get('status') != 'ok' for r in rows)


def write(state):
    state['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    path = SUITE / 'resume_status.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2)+'\n')
    tmp.replace(path)


def main():
    SUITE.mkdir(parents=True, exist_ok=True)
    lock = (SUITE / 'resume.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, BENCHMARK_PYTHON=sys.executable,
        BENCHMARK_SUFFIX='merged_layers_per_face', PYTHONUNBUFFERED='1',
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    state = dict(supervisor_pid=os.getpid(), state='running',
                 alignment='all-chamfer', datasets={})
    for dataset in ('sage10k', '3dfront'):
        out = BASE / f'{dataset}_independent_faces_cube90_merged_layers_per_face'
        out.mkdir(exist_ok=True)
        result = out / 'results.json'
        if result.exists():
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            shutil.copy2(result, out / f'results.before_resume_{stamp}.json')
        for attempt in range(1, 4):
            successful, failed = counts(dataset)
            if successful == 100 and failed == 0:
                state['datasets'][dataset] = dict(state='complete', successful=100, failed=0)
                write(state)
                break
            command = ['bash', str(ROOT / 'scripts/eval/benchmark_single_image.sh'),
                       'lari', dataset, '--lari-alignment', 'all-chamfer']
            with (SUITE / 'resume.log').open('a') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                record = dict(state='running', pid=child.pid, attempt=attempt,
                              successful=successful, failed=failed, command=command)
                state['datasets'][dataset] = record
                write(state)
                print(f'{dataset}: resuming {successful}/100; launcher PID {child.pid}', flush=True)
                while True:
                    successful, failed = counts(dataset)
                    record.update(successful=successful, failed=failed)
                    write(state)
                    try:
                        code = child.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            successful, failed = counts(dataset)
            record.update(state='complete' if successful == 100 and failed == 0 else 'incomplete',
                          successful=successful, failed=failed, returncode=code)
            write(state)
            if record['state'] == 'complete':
                break
            time.sleep(10)
    state['state'] = 'complete' if all(
        r.get('state') == 'complete' for r in state['datasets'].values()) else 'incomplete'
    write(state)


if __name__ == '__main__':
    main()
