"""Recoverable training-first queue, with independent shell watchdog and idle GPU work."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/opt/conda/envs/nova3r/bin/python'
QUEUE = ROOT / 'exp_output/hunyuandit_panorama_no_plucker/supervised_queue'
BASE = ROOT / 'exp_output/hunyuandit_panorama/meshudf_apd_resolution4'
RUNS = {'scannetpp': BASE / 'resolution1024_reference512_full', '3dfront': BASE / '3dfront_resolution1024_reference512_steps100_full'}

def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default

def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)

def process(pid):
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0], fields[19]
    except (OSError, ValueError, TypeError):
        return None

def live(record):
    info = process(record.get('pid')) if record else None
    return bool(info and info[0] != 'Z' and info[1] == record.get('start_ticks'))

def running(name, record):
    if live(record):
        return True
    child = read(QUEUE / f'{name}_exit.json', {})
    return bool(record and child.get('started', 0) >= record['started'] and live(child))

def spawn(command, log, env=None):
    with log.open('a') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    info = process(child.pid)
    return dict(pid=child.pid, start_ticks=info[1], started=time.time())

def stop(record):
    if not live(record):
        return
    os.killpg(record['pid'], signal.SIGCONT)
    os.killpg(record['pid'], signal.SIGTERM)
    for _ in range(10):
        if not live(record):
            return
        time.sleep(1)
    if live(record):
        os.killpg(record['pid'], signal.SIGKILL)

def completed(dataset):
    out = RUNS[dataset]
    listfile = BASE / 'full_list.json' if dataset == 'scannetpp' else out / 'full_list.json'
    expected = set()
    for item in read(listfile, []):
        if dataset == '3dfront' and '/' in item:
            parts = Path(item).parts
            frame = int(Path(parts[-1]).stem.rsplit('_', 1)[1])
            expected.add(f'3dfront_{parts[0]}_{parts[1]}_{frame:05d}_4')
        else:
            expected.add(dataset + '_' + item.replace(' ', '_') + '_4')
    good = set()
    for path in out.rglob('metrics.json'):
        row = read(path, {})
        if row.get('status') == 'ok' and Path(row.get('mesh', '')).is_file():
            good.add(row['sample'])
    return len(good & expected), len(expected)

def progress(name, record):
    latest = record['started']
    paths = [QUEUE / f'{name}.log']
    if name == 'training':
        paths += list((ROOT / 'exp_output/hunyuandit_panorama_no_plucker/nova3r_training').glob('version_*/events.out.tfevents.*'))
        paths += list((ROOT / 'exp_output/hunyuandit_panorama_no_plucker/checkpoints').glob('*.ckpt'))
    elif name in RUNS:
        paths += list(RUNS[name].rglob('metrics.json'))
    for path in paths:
        try:
            latest = max(latest, path.stat().st_mtime)
        except OSError:
            pass
    return latest

def launch(name, state):
    env = dict(CUDA_VISIBLE_DEVICES='0', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8', MPLCONFIGDIR='/tmp/nova3r-matplotlib')
    if name == 'training':
        command = [PYTHON, '-u', str(ROOT / 'scripts/training/resume_guarded.py')]
    else:
        out = RUNS[name]
        command = read(out / 'launch.json')['command']
        env.update(MESH_UDF_SEARCH_ROOT=str(BASE if name == 'scannetpp' else out), MESH_UDF_RESUME_COMPLETED=str(out))
    spec = dict(command=command, cwd=str(ROOT), env=env, log=str(QUEUE / f'{name}.log'), status=str(QUEUE / f'{name}_exit.json'))
    specpath = QUEUE / f'{name}_spec.json'
    write(specpath, spec)
    logfile = Path(spec['log'])
    offset = logfile.stat().st_size if logfile.exists() else 0
    record = spawn([PYTHON, '-u', str(ROOT / 'scripts/training/guarded_job.py'), str(specpath)], QUEUE / 'wrapper.log')
    record['log_offset'] = offset
    record['attempt'] = state['jobs'].get(name, {}).get('attempt', 0) + 1
    state['jobs'][name] = record
    print(f'Started {name}, attempt {record["attempt"]}, wrapper PID {record["pid"]}', flush=True)

def main():
    QUEUE.mkdir(parents=True, exist_ok=True)
    lock = (QUEUE / 'supervisor.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    state = read(QUEUE / 'state.json', {'jobs': {}, 'phase': 'training'})
    state['supervisor_pid'] = os.getpid()
    setup = read(QUEUE / 'setup_keepalive.json', {})
    if setup:
        try:
            args = Path(f'/proc/{setup["pid"]}/cmdline').read_bytes()
            if b'gpu_keepalive.py' in args:
                os.kill(setup['pid'], signal.SIGTERM)
        except OSError:
            pass
    while True:
        # The shell restarts us; we also restore the shell if it is killed.
        guard = read(QUEUE / 'watchdog.json', {})
        if not live(guard):
            spawn(['bash', str(ROOT / 'scripts/training/guarded_queue_watchdog.sh')], QUEUE / 'watchdog.log')
        elif process(guard['pid'])[0] == 'T':
            os.kill(guard['pid'], signal.SIGCONT)
        if (QUEUE / 'PAUSE').exists():
            for name, record in state['jobs'].items():
                if running(name, record) and not live(record):
                    os.killpg(record['pid'], signal.SIGKILL)
                else:
                    stop(record)
            state['phase'] = 'paused'
            desired = []
        elif not (QUEUE / 'training_complete.json').exists():
            state['phase'] = 'training'
            desired = ['training']
        else:
            state['counts'] = {key: completed(key) for key in RUNS}
            desired = [key for key, (done, expected) in state['counts'].items() if not expected or done < expected]
            state['phase'] = 'evaluations' if desired else 'complete_keepalive'
        for name in desired:
            record = state['jobs'].get(name)
            if running(name, record):
                # Resume externally SIGSTOP'ed process groups, including loaders.
                for pid in [record['pid'], read(QUEUE / f'{name}_exit.json', {}).get('pid')]:
                    info = process(pid)
                    if info and info[0] == 'T':
                        os.killpg(record['pid'], signal.SIGCONT)
                timeout = 7200 if name == 'training' else 3600
                if time.time() - progress(name, record) > timeout:
                    print(f'{name} stalled; restarting from saved progress', flush=True)
                    if live(record):
                        stop(record)
                    else:
                        os.killpg(record['pid'], signal.SIGKILL)
                else:
                    continue
            if record and 'retry_after' not in record:
                record['retry_after'] = time.time() + min(300, 30 * record.get('attempt', 1))
                print(f'{name} exited; retry after {record["retry_after"]}', flush=True)
            if record and time.time() < record.get('retry_after', 0):
                continue
            keep = state.get('keepalive')
            stop(keep)
            state['keepalive'] = None
            launch(name, state)
            write(QUEUE / 'state.json', state)
        busy = any(running(name, state['jobs'].get(name)) for name in desired)
        initializing = False
        if state['phase'] == 'training' and busy:
            log = QUEUE / 'training.log'
            # Check only this attempt, so an earlier successful restore cannot
            # suppress keep-alive during a later retry's long CPU load.
            record = state['jobs']['training']
            restored = record.get('restored', False)
            if not restored and log.exists():
                exit_info = read(QUEUE / 'training_exit.json', {})
                if exit_info.get('started', 0) >= record['started']:
                    with log.open() as stream:
                        stream.seek(record.get('log_offset', 0))
                        restored = 'Restored all states from the checkpoint' in stream.read()
                    record['restored'] = restored
            initializing = not restored
        if (not busy or initializing) and not live(state.get('keepalive')):
            state['keepalive'] = spawn([PYTHON, '-u', str(ROOT / 'scripts/training/gpu_keepalive.py'), '--memory-fraction', '0.05'], QUEUE / 'keepalive.log')
        elif busy and not initializing and live(state.get('keepalive')):
            stop(state['keepalive'])
            state['keepalive'] = None
        # Reap only our own exited children; adopted wrappers remain monitored.
        try:
            while os.waitpid(-1, os.WNOHANG)[0]:
                pass
        except ChildProcessError:
            pass
        state['heartbeat_utc'] = time.time()
        write(QUEUE / 'state.json', state)
        time.sleep(10)

if __name__ == '__main__':
    main()
