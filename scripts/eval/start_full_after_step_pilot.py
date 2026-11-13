"""After three paired step comparisons, launch resumable full ScanNet1024."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'exp_output/hunyuandit_panorama'
PILOT = BASE/'flow_steps_1024_generalization'
FULL = BASE/'meshudf_apd_resolution4'
STEPS = (50,100,200)
METRICS = ('visible_cd','visible_apd_0.1','visible_apd_0.05','occluded_cd',
           'occluded_apd_0.1','occluded_apd_0.05','complete_cd','f_score_0.1_full','f_score_0.05_full')


def state(value, **extra):
    record=dict(state=value,step_counts=list(STEPS),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
    (PILOT/'auto_full_status.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


def read(path):
    try: return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError): return None


def complete_pairs():
    good=[]
    for path in PILOT.glob('*/paired/*.json'):
        record=read(path)
        if not record: continue
        variants={v['steps']:v for v in record['variants']}
        if set(STEPS).issubset(variants) and all(len(variants[s]['results'])==1 and variants[s]['results'][0].get('status')=='ok' for s in STEPS):
            good.append((path.stat().st_mtime,record))
    return [r for _,r in sorted(good)[:3]]


def select_steps(pairs):
    if len(pairs)!=3:
        raise ValueError('Exactly three complete scene comparisons are required')
    averages={}
    for steps in STEPS:
        records=[next(v for v in p['variants'] if v['steps']==steps)['results'][0] for p in pairs]
        averages[steps]={k:sum(r[k] for r in records)/3 for k in METRICS}
    selected=max(STEPS,key=lambda s:(averages[s]['occluded_apd_0.05'],-averages[s]['occluded_cd'],-s))
    return selected,averages


def terminate(pid, marker):
    path=Path('/proc')/str(pid)
    try:
        if marker.encode() not in (path/'cmdline').read_bytes(): return
        # A paused worker cannot process SIGTERM; continue only to terminate.
        status=(path/'stat').read_text().rsplit(')',1)[1].split()[0]
        if status=='T': os.kill(pid,signal.SIGKILL)
        else:
            os.kill(pid,signal.SIGTERM)
            time.sleep(2)
            if path.exists() and marker.encode() in (path/'cmdline').read_bytes():
                os.kill(pid,signal.SIGKILL)
    except (FileNotFoundError,ProcessLookupError): pass


def main():
    state('waiting_for_three_scenes_and_memory_checks')
    last_progress=None
    while True:
        pairs=complete_pairs()
        cpu=read(PILOT/'memory_safety/stress_1024.json')
        gpu=read(PILOT/'memory_safety/gpu_probe.json')
        progress=(len(pairs),bool(cpu and cpu.get('status')=='ok'),bool(gpu and gpu.get('status')=='ok'))
        if progress!=last_progress:
            state('waiting_for_three_scenes_and_memory_checks',completed_scenes=progress[0],required_scenes=3,
                  cpu_stress_passed=progress[1],gpu_query_check_passed=progress[2])
            last_progress=progress
        if len(pairs)==3 and cpu and gpu and cpu.get('status')==gpu.get('status')=='ok': break
        status=read(PILOT/'status.json') or {}
        if status.get('state')=='failed':
            state('failed',reason='Step pilot failed before three complete scenes',pilot=status)
            return
        time.sleep(10)
    selected,averages=select_steps(pairs)
    selection=dict(selected_steps=selected,samples=[p['sample'] for p in pairs],metrics=averages,
        criterion='Highest mean occluded APD5 over three complete scenes; lower occluded CD and then fewer steps break ties.',
        validation_scope='Three-scene pilot; dataset-wide performance remains to be measured.',memory_checks=dict(cpu=cpu,gpu=gpu))
    (PILOT/'three_scene_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    # Stop the remaining pilot so it does not contend with the selected full run.
    launch=read(PILOT/'launch.json') or {}
    status=read(PILOT/'status.json') or {}
    if launch.get('pid'): terminate(launch['pid'],'run_flow_steps_benchmark.py')
    if status.get('pid'): terminate(status['pid'],'diagnose_meshudf_resolution.py')
    (PILOT/'status.json').write_text(json.dumps(dict(state='stopped_after_three_complete_scenes',selected_steps=selected),indent=2)+'\n')
    previous=FULL/'resolution1024_reference512_full'
    old=read(previous/'launch.json') or {}
    if old.get('pid'): terminate(old['pid'],'sweep_meshudf_occluded.py')
    pending=read(PILOT/'pending_previous_full.json') or {}
    if pending.get('pid'): terminate(pending['pid'],'sweep_meshudf_occluded.py')
    out=FULL/f'resolution1024_reference512_steps{selected}_full'
    out.mkdir(exist_ok=True)
    (previous/'SUPERSEDED.json').write_text(json.dumps(dict(reason='Restarted with the selected step count and memory safeguards',output=str(out)),indent=2)+'\n')
    cfg=yaml.safe_load((previous/'config.yaml').read_text())
    cfg['cfg'].update(test_flow_matching_steps=selected,output_dir=str(out),
        mesh_udf_query_batch_size=65536,mesh_udf_vertex_batch_size=32768,
        mesh_udf_streamed_grid=True,mesh_udf_mc_tile_cells=0)
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',MESH_UDF_SEARCH_ROOT=str(FULL),MESH_UDF_RESUME_COMPLETED=str(out))
    for attempt in range(1,4):
        (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        command=[sys.executable,'-u',str(ROOT/'scripts/eval/sweep_meshudf_occluded.py'),'worker',
                 '--config',str(out/'config.yaml'),'--devices','1','--low-cpu-mem-usage',
                 '--test-dataset','scannetpp','--test-list',str(FULL/'full_list.json'),
                 '--mesh-method','mesh_udf','--output-dir',str(out)]
        with (out/f'evaluation_attempt_{attempt}.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            (out/'launch.json').write_text(json.dumps(dict(pid=child.pid,command=command,attempt=attempt,selection=selection),indent=2)+'\n')
            state('full_run_running',selected_steps=selected,pid=child.pid,attempt=attempt,output=str(out))
            with (out/f'comparison_monitor_attempt_{attempt}.log').open('w') as monitor_log:
                monitor=subprocess.Popen([sys.executable,str(ROOT/'scripts/eval/report_meshudf_resolution.py'),
                    str(FULL/'resolution512_full'),str(out),'--output',str(out/'comparison_to_512'),
                    '--watch-pid',str(child.pid)],cwd=ROOT,stdout=monitor_log,stderr=subprocess.STDOUT)
            code=child.wait()
        records={}
        for path in sorted(out.rglob('metrics.json'),key=lambda p:p.stat().st_mtime):
            record=read(path)
            if record: records[record['sample']]=record
        successful=sum(r.get('status')=='ok' for r in records.values())
        if code==0 and successful==100:
            state('complete',selected_steps=selected,successful=successful,output=str(out))
            return
        state('retrying_full_run',attempt=attempt,returncode=code,successful=successful,output=str(out))
        cfg['cfg']['mesh_udf_query_batch_size']=max(4096,cfg['cfg']['mesh_udf_query_batch_size']//2)
        cfg['cfg']['mesh_udf_vertex_batch_size']=max(4096,cfg['cfg']['mesh_udf_vertex_batch_size']//2)
        cfg['cfg']['mesh_udf_mc_tile_cells']=64
    state('failed',reason='Full evaluation remained incomplete after three resumable attempts',output=str(out))


if __name__=='__main__': main()
