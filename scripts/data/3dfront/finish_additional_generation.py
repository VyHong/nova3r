#!/usr/bin/env python3
"""Finish 1,030 successful new rooms and update backed-up train/test splits."""
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[3]
os.chdir(ROOT)
STATE = ROOT / 'artifacts/3dfront_extension'
RENDERS = ROOT / 'artifacts/3dfront_additional_1000'
DATA = ROOT / 'data/3dfront'
SPEC = importlib.util.spec_from_file_location('renderer', ROOT / 'scripts/data/3dfront/render_rooms_batched.py')
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def key(job):
    return Path(job['scene']).stem, job['room']


def folder(job):
    scene, room = key(job)
    return RENDERS / scene / room


def complete(job):
    try:
        return renderer.complete(folder(job), job['room'], 10, 1036, 64)
    except (OSError, ValueError, KeyError):
        return False


def entries(job):
    scene, room = key(job)
    count = read(folder(job) / 'metadata.json')['view_count']
    poses = read(folder(job) / 'extrinsics.json')
    result = [f'{scene}/{room}/{room}_panorama_{i:03d}.png' for i in range(count)]
    assert all(Path(p).name in poses for p in result), job
    return result


def dwelling(entry):
    return Path(entry).parts[0]


def main():
    STATE.mkdir(exist_ok=True)
    backup = STATE / 'original_splits'
    backup.mkdir(exist_ok=True)
    names = ['train_list', 'test_list', 'final_test_list', 'final_test_10_images']
    for name in names:
        dest = backup / (name + '.json')
        if not dest.exists():
            shutil.copy2(DATA / dest.name, dest)
    original = {name: read(backup / (name + '.json')) for name in names}
    jobs = read(ROOT / 'artifacts/3dfront_additional_1000_manifest.json')
    original_keys = {key(j) for j in jobs}
    if (STATE / 'successful_1030.json').exists():
        jobs += read(STATE / 'successful_1030.json')
    if (STATE / 'attempted.json').exists():
        jobs += read(STATE / 'attempted.json')
    unique = {key(j): j for j in jobs}
    jobs = list(unique.values())
    occupied = {key(j)[0] for j in jobs}
    heldout = {dwelling(e) for e in original['test_list'] + original['final_test_list']}
    candidates = []
    for job in read(STATE / 'candidates.json'):
        scene, _ = key(job)
        if scene not in occupied and scene not in heldout:
            candidates.append(job)
            occupied.add(scene)
    successes = [j for j in jobs if complete(j)]
    cursor = 0
    while len(successes) < 1030:
        missing = 1030 - len(successes)
        batch = candidates[cursor:cursor + missing]
        cursor += len(batch)
        if not batch:
            raise RuntimeError(f'Candidate pool exhausted at {len(successes)} successes; splits unchanged')
        write(STATE / 'batch.json', batch)
        jobs += batch
        write(STATE / 'attempted.json', [j for j in jobs if key(j) not in original_keys])
        print(f'{len(successes)} successful; rendering {len(batch)} replacement rooms', flush=True)
        command = [sys.executable, '-u', str(ROOT / 'scripts/data/3dfront/render_rooms_batched.py'),
                   '--manifest', str(STATE / 'batch.json'), '--output', str(RENDERS),
                   '--adaptive-views', '--views', '10', '--samples', '64', '--width', '1036',
                   '--workers', '10', '--device', 'GPU', '--gpu-backend', 'OPTIX',
                   '--denoiser', 'OPTIX', '--gpu-ids', '0']
        subprocess.run(command, check=True)
        successes = [j for j in jobs if complete(j)]
    assert len(successes) == 1030
    train_dwellings = {dwelling(e) for e in original['train_list']}
    groups = [({'Hallway','Corridor','Aisle','Stairwell'},12),
              ({'CloakRoom','StorageRoom','EquipmentRoom','LaundryRoom','Garage','BathRoom','Bathroom','MasterBathroom','SecondBathroom','Kitchen'},6),
              ({'BedRoom','ElderlyRoom','NannyRoom'},4),
              ({'Library','Lounge','Auditorium','OtherRoom','OtherSpace'},5),
              ({'Balcony','Courtyard','Terrace'},3)]
    test_jobs = []
    chosen = set()
    for kinds, target in groups:
        pools = defaultdict(list)
        for job in successes:
            if job['room_type'] in kinds and key(job)[0] not in train_dwellings:
                pools[job['room_type']].append(job)
        selected = 0
        while selected < target:
            progressed = False
            for kind in sorted(pools):
                pool = pools[kind]
                while pool and key(pool[0]) in chosen:
                    pool.pop(0)
                if pool and selected < target:
                    job = pool.pop(0)
                    test_jobs.append(job)
                    chosen.add(key(job))
                    selected += 1
                    progressed = True
            if not progressed:
                raise RuntimeError(f'Insufficient test rooms for {kinds}; splits unchanged')
    test_dwellings = {key(j)[0] for j in test_jobs}
    train_jobs = [j for j in successes if key(j)[0] not in test_dwellings]
    assert len(train_jobs) == 1000 and len(test_jobs) == 30
    added_train = [e for j in train_jobs for e in entries(j)]
    new_train = original['train_list'] + added_train
    # Preserve the existing ten-example subset and remove 30 other old test rooms.
    protected = {str(Path(e).parent) for e in original['final_test_10_images']}
    eligible = [e for e in original['final_test_list'] if str(Path(e).parent) not in protected]
    removed = random.Random(42).sample(eligible, 30)
    removed_rooms = {str(Path(e).parent) for e in removed}
    new_examples = [entries(j)[len(entries(j)) // 2] for j in test_jobs]
    new_final = [e for e in original['final_test_list'] if e not in removed] + new_examples
    new_test = [e for e in original['test_list'] if str(Path(e).parent) not in removed_rooms]
    new_test += [e for j in test_jobs for e in entries(j)]
    assert len(new_final) == 100 and len(set(new_final)) == 100
    assert len({str(Path(e).parent) for e in new_train}) == 3000
    assert not ({dwelling(e) for e in new_train} & {dwelling(e) for e in new_test})
    assert set(original['final_test_10_images']) <= set(new_final)
    for j in successes:
        entries(j)
    # All assertions precede split writes and quarantine operations.
    write(STATE / 'successful_1030.json', successes)
    write(STATE / 'train_1000.json', train_jobs)
    write(STATE / 'test_30.json', test_jobs)
    write(DATA / 'additional_train_list.json', added_train)
    write(DATA / 'additional_test_list.json', [e for j in test_jobs for e in entries(j)])
    write(DATA / 'train_list.json', new_train)
    write(DATA / 'test_list.json', new_test)
    write(DATA / 'final_test_list.json', new_final)
    write(STATE / 'replaced_test_examples.json', removed)
    quarantined = []
    for job in jobs:
        if complete(job):
            continue
        src = folder(job)
        if src.exists():
            dest = STATE / 'failed_rooms' / key(job)[0] / key(job)[1]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise FileExistsError(dest)
            shutil.move(str(src), str(dest))
            quarantined.append(str(dest.relative_to(ROOT)))
    report = {'successful_rooms':len(successes),'new_train_rooms':len(train_jobs),
              'new_test_rooms':len(test_jobs),'train_images':len(new_train),
              'final_test_images':len(new_final),'test_images':len(new_test),
              'quarantined':quarantined,'test_room_types':dict(Counter(j['room_type'] for j in test_jobs))}
    write(STATE / 'summary.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
