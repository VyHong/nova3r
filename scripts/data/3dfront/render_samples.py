#!/usr/bin/env python3
"""Render ten furnished, windowed 3D-FRONT rooms, with resumable outputs."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / 'artifacts/3dfront_gpu_samples'
OUTPUT.mkdir(parents=True, exist_ok=True)
manifest_path = OUTPUT / 'manifest.json'
if manifest_path.exists():
    jobs = json.loads(manifest_path.read_text())
else:
    jobs = []
    for scene in sorted(Path('/tmp/datasets/3dfront/3D-FRONT').glob('*.json')):
        data = json.loads(scene.read_text())
        meshes = {m['uid']: m for m in data['mesh']}
        for room in data['scene']['room']:
            children = room.get('children', [])
            types = {meshes[c['ref']].get('type', '').lower() for c in children if c['ref'] in meshes}
            furnished = sum(c['ref'] not in meshes for c in children)
            if {'window', 'floor'} <= types and furnished >= 4 and room.get('type') in {'LivingDiningRoom', 'LivingRoom', 'MasterBedroom', 'SecondBedroom', 'Bedroom', 'DiningRoom', 'Study', 'KidsRoom'}:
                jobs.append({'scene': str(scene), 'room': room['instanceid'], 'output': str(OUTPUT / f'{len(jobs)+1:02d}_{room["instanceid"]}')})
                break
        if len(jobs) == 10:
            break
    if len(jobs) != 10:
        raise RuntimeError('Could not find ten eligible rooms')
    manifest_path.write_text(json.dumps(jobs, indent=2) + '\n')
for index, job in enumerate(jobs):
    folder = Path(job['output'])
    folder.mkdir(parents=True, exist_ok=True)
    hdri_name = ['kiara_3_morning', 'canary_wharf', 'kiara_7_late-afternoon',
                 'urban_courtyard', 'kiara_5_noon', 'modern_buildings',
                 'kiara_3_morning', 'modern_buildings_night', 'moonlit_golf',
                 'rooftop_night'][index]
    hdri = next(Path('/tmp/datasets/polyhaven/hdris', hdri_name).glob('*.hdr'))
    ceiling_strength = [0.35, 0.55, 0.8, 1.1, 1.4, 0.65, 1.0, 0.5, 0.8, 1.1][index]
    lamp_strength = [4.0, 6.0, 8.0, 10.0, 12.0, 7.0, 9.0, 12.0, 16.0, 20.0][index]
    def is_complete():
        if not all((folder / name).is_file() and (folder / name).stat().st_size > 0 for name in ['metadata.json', 'panorama.png', 'room.glb']):
            return False
        metadata = json.loads((folder / 'metadata.json').read_text())
        return (metadata.get('door_selection_version') == 2
                and metadata.get('window_geometry_version') == 2
                and metadata.get('door_geometry_version') == 2
                and metadata.get('lighting_version') == 3
                and metadata.get('hdri') == str(hdri)
                and metadata.get('ceiling_emission_strength') == ceiling_strength
                and metadata.get('lamp_emission_strength') == lamp_strength)
    if is_complete():
        continue
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'third_party/BlenderProc'), FRONT3D_SCENE=job['scene'], FRONT3D_ROOM=job['room'], FRONT3D_OUTPUT=job['output'], FRONT3D_HDRI=str(hdri), FRONT3D_HDRI_ROTATION=str(index * 37), FRONT3D_HDRI_STRENGTH='0.85', FRONT3D_CEILING_EMISSION_STRENGTH=str(ceiling_strength), FRONT3D_LAMP_EMISSION_STRENGTH=str(lamp_strength))
    env['FRONT3D_TIME_OF_DAY'] = 'night' if index >= 7 else 'day'
    env['FRONT3D_ENVIRONMENT'] = 'city' if index % 2 else 'landscape'
    env['FRONT3D_WINDOW_STYLE'] = ['picture', 'double_casement', 'triple_panel', 'grid', 'transom', 'awning', 'arched_transom'][index % 7]
    env['FRONT3D_DOOR_STYLE'] = ['flush', 'two_panel', 'six_panel', 'glazed', 'louvered'][index % 5]
    print(f'Rendering {index+1}/10: {job["room"]}', flush=True)
    with (folder / 'render.log').open('w') as log:
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/data/3dfront/render_front3d_scene.py')], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if not is_complete():
        print((folder / 'render.log').read_text()[-6000:], flush=True)
        raise SystemExit(result.returncode or 1)
    if result.returncode:
        print(f'Blender exited with {result.returncode} during shutdown; all outputs were saved.', flush=True)
print('Completed all ten panoramas and GLBs.', flush=True)
