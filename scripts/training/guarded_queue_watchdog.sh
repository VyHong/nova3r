#!/usr/bin/env bash
set -u
cd /mnt/home/vyhong/projects/nova3r
queue=exp_output/hunyuandit_panorama_no_plucker/supervised_queue
mkdir -p "$queue"
exec 9>"$queue/watchdog.lock"
flock -n 9 || exit 0
trap '' HUP
/opt/conda/envs/nova3r/bin/python - "$queue/watchdog.json" "$$" <<'PY'
import json,os,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);pid=int(sys.argv[2])
ticks=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(pid=pid,start_ticks=ticks,started=time.time())));os.replace(t,p)
PY
check_supervisor() {
    /opt/conda/envs/nova3r/bin/python - "$queue/state.json" <<'PY'
import json,os,pathlib,signal,sys,time
try:
    state=json.loads(pathlib.Path(sys.argv[1]).read_text())
    pid=state['supervisor_pid']
    command=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
    if b'guarded_queue.py' in command:
        os.kill(pid,signal.SIGCONT)
        if time.time()-state.get('heartbeat_utc',time.time())>180:
            os.kill(pid,signal.SIGKILL)
except (OSError,ValueError,KeyError):
    pass
PY
}
while true; do
    check_supervisor
    /opt/conda/envs/nova3r/bin/python -u scripts/training/guarded_queue.py 9>&- >> "$queue/supervisor.log" 2>&1 &
    supervisor_pid=$!
    while kill -0 "$supervisor_pid" 2>/dev/null; do
        check_supervisor
        kill -CONT "$supervisor_pid" 2>/dev/null || true
        sleep 10
    done
    wait "$supervisor_pid" || true
    sleep 10
done
