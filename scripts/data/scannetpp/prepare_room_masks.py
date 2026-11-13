"""Copy missing scan masks and remap anonymized vertices to cropped room meshes."""
import argparse
import json
from pathlib import Path
import shutil
import warnings

import numpy as np
from scipy.spatial import cKDTree
import trimesh


def read_indices(path, count):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        indices = np.loadtxt(path, dtype=np.int64, ndmin=1).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= count):
        raise ValueError(f'Out-of-range vertex indices: {path}')
    return indices


def prepare(root, source_root=None):
    copied = 0
    for mesh in sorted(root.glob('*/scans/mesh_aligned_0.05.ply')):
        mask = mesh.with_name('mesh_aligned_0.05_mask.txt')
        if not mask.is_file():
            source = source_root / mesh.parent.parent.name / 'scans' / mask.name if source_root else None
            if source is None or not source.is_file():
                raise FileNotFoundError(f'Missing {mask}; provide --source-root containing original scan masks')
            shutil.copy2(source, mask)
            copied += 1
    records = []
    for scene in sorted((root / 'rooms').iterdir()):
        rooms = sorted(scene.glob('*/mesh.ply'))
        if not rooms:
            continue
        original = root / scene.name / 'scans' / 'mesh_aligned_0.05.ply'
        vertices = np.asarray(trimesh.load(original, process=False).vertices)
        source_mask = original.with_name('mesh_aligned_0.05_mask.txt')
        indices = read_indices(source_mask, len(vertices))
        tree = cKDTree(vertices[indices]) if len(indices) else None
        for path in rooms:
            room = trimesh.load(path, process=False)
            local = np.flatnonzero(tree.query(room.vertices)[0] <= 1e-5) if tree else np.empty(0, dtype=np.int64)
            mask = path.with_name('mesh_mask.txt')
            np.savetxt(mask, local, fmt='%d')
            excluded = np.zeros(len(room.vertices), dtype=bool)
            excluded[local] = True
            record = dict(room=str(path.parent), source_mask=str(source_mask),
                          mask=str(mask), vertices=len(room.vertices),
                          masked_vertices=len(local), masked_faces=int(excluded[np.asarray(room.faces)].any(axis=1).sum()),
                          coordinate_tolerance=1e-5)
            mask.with_suffix('.json').write_text(json.dumps(record, indent=2)+'\n')
            records.append(record)
            print(f'{path.parent.name}: {len(local)} masked vertices', flush=True)
    print(f'Copied {copied} missing scan masks; prepared {len(records)} room masks.', flush=True)
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('datasets/scannetpp/test_scenes'))
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args()
    prepare(args.root, args.source_root)
