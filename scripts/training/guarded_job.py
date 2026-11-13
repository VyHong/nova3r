"""Persistent job wrapper: record child identity and exit status for adoption."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)

if __name__ == '__main__':
    spec = json.loads(Path(sys.argv[1]).read_text())
    status = Path(spec['status'])
    with Path(spec['log']).open('a') as log:
        child = subprocess.Popen(spec['command'], cwd=spec['cwd'], env=dict(os.environ, **spec['env']), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
        write(status, dict(wrapper_pid=os.getpid(), pid=child.pid, start_ticks=ticks, started=time.time(), command=spec['command']))
        code = child.wait()
        write(status, dict(wrapper_pid=os.getpid(), pid=child.pid, returncode=code, finished=time.time(), command=spec['command']))
        sys.exit(code)
