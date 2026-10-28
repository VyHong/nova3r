import json
import shutil
from collections import defaultdict
from pathlib import Path
import numpy as np

root = Path('datasets/scannetpp/test_scenes')
pose_root = Path('/mnt/projects/theses/vyhong/scannetpp/scan_poses')
image_root = Path('/tmp/datasets/scannetpp/cubemap_data')
frames = defaultdict(list)
for entry in json.loads(Path('data/scannetpp/final_test_list.json').read_text()):
    scene, frame = entry.split()
    frames[scene].append(frame)
basis = np.array([[0,1,0,0],[0,0,-1,0],[-1,0,0,0],[0,0,0,1]], dtype=float)
outputs = []
for scene, ids in sorted(frames.items()):
    folder = root / scene
    folder.mkdir(parents=True, exist_ok=True)
    source = pose_root / scene / 'scans/scanner_poses.json'
    poses = json.loads(source.read_text())
    selected = {}
    for frame in [f"{index:05d}" for index in range(len(poses))]:
        scanner_c2w = np.asarray(poses[int(frame)], dtype=float)
        assert scanner_c2w.shape == (4,4) and np.isfinite(scanner_c2w).all()
        camera_path = image_root / scene / 'panocam/images' / f'{int(frame)}_cube_95/camera_data.json'
        data = json.loads(camera_path.read_text()) if camera_path.is_file() else {}
        faces = {}
        for face in range(6) if data else []:
            camera = data[f'{face:04d}']
            # Dataset: face_w2c @ (B @ scanner_w2c @ B.T) @ B.
            w2c = np.asarray(camera['extrinsics']) @ basis @ np.linalg.inv(scanner_c2w)
            c2w = np.linalg.inv(w2c)
            np.testing.assert_allclose(w2c @ c2w, np.eye(4), atol=1e-8)
            faces[f'{face:04d}'] = {
                'world_to_camera_opencv': w2c.tolist(),
                'camera_to_world_opencv': c2w.tolist(),
                'intrinsics_normalized': camera['intrinsics'],
            }
        selected[frame] = {'scanner_camera_to_world': scanner_c2w.tolist(), 'cubemap_faces': faces,
                           'is_test': frame in ids}
    payload = {
        'scene': scene,
        'test_list': 'data/scannetpp/final_test_list.json',
        'world_frame': 'Original scan mesh coordinates (scans/mesh_aligned_0.05.ply), without training normalization.',
        'scanner_pose_convention': 'Original scanner camera-to-world matrix, indexed by panorama frame ID.',
        'cubemap_pose_convention': 'OpenCV camera axes: X right, Y down, Z forward. Matrices act on homogeneous column vectors.',
        'intrinsics_convention': 'Normalized intrinsics, as stored in cubemap camera_data.json; scale rows 0/1 by image width/height for pixels.',
        'frames': selected,
    }
    outputs.append((source, folder / 'scans/scanner_poses.json', folder / 'test_camera_poses.json', payload))
# Resolve and validate every input before writing any output.
for source, raw_dest, dest, payload in outputs:
    if raw_dest.exists() and raw_dest.read_bytes() != source.read_bytes():
        raise RuntimeError(f'Existing different file: {raw_dest}')

for source, raw_dest, dest, payload in outputs:
    raw_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, raw_dest)
    dest.write_text(json.dumps(payload, indent=2) + '\n')
    assert json.loads(dest.read_text()) == payload
print(f'Exported all scanner cameras for {len(outputs)} scenes, marking {sum(map(len, frames.values()))} test frames.')
