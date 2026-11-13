"""Create a cleaned, Taubin-smoothed mesh copy for qualitative rendering."""
import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--iterations', type=int, default=30)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('--iterations must be positive')
    output = args.output or args.mesh.with_name(args.mesh.stem+'_render_smoothed.ply')
    if output.resolve() == args.mesh.resolve():
        parser.error('Output must differ from the original mesh')
    mesh = o3d.io.read_triangle_mesh(str(args.mesh))
    if mesh.is_empty() or not mesh.has_triangles():
        parser.error('Expected a triangle mesh')
    before = {'vertices':len(mesh.vertices),'triangles':len(mesh.triangles)}
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices).copy()
    print('Cleaned:',before,'->',len(mesh.vertices),'vertices,',len(mesh.triangles),'triangles',flush=True)
    # Alternating positive/negative Laplacian steps reduce shrinkage compared
    # with ordinary Laplacian smoothing. No hole filling or remeshing.
    mesh = mesh.filter_smooth_taubin(number_of_iterations=args.iterations,lambda_filter=0.5,mu=-0.53)
    mesh.compute_vertex_normals()
    displacement = np.linalg.norm(np.asarray(mesh.vertices)-vertices,axis=1)
    output.parent.mkdir(parents=True,exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output),mesh):
        raise RuntimeError(f'Failed to write {output}')
    report = {
        'source':str(args.mesh),'output':str(output),'before':before,
        'after':{'vertices':len(mesh.vertices),'triangles':len(mesh.triangles)},
        'method':'Taubin','iterations':args.iterations,'lambda':0.5,'mu':-0.53,
        'displacement_mesh_units':{'mean':float(displacement.mean()),'p95':float(np.percentile(displacement,95)),'max':float(displacement.max())},
        'source_modified':False,'remeshing':False,'decimation':False,'hole_filling':False,
    }
    output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
