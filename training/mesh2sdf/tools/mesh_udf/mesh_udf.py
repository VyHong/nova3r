import argparse
import contextlib
import torch
import numpy as np
from scipy.sparse import coo_matrix
import trimesh
from torch.nn import functional as F
import sys
import os
import tqdm
from collections import defaultdict

from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes

import training.mesh2sdf.tools.mesh_udf.workspace as ws

import sys
sys.path.append('custom_mc')
from training.mesh2sdf.tools.mesh_udf.custom_mc._marching_cubes_lewiner import udf_mc_lewiner


def _make_decoder_input(latent_vec, xyz):
    xyz = xyz.reshape(-1, xyz.shape[-1])
    if latent_vec is None:
        return xyz

    latent_dim = latent_vec.shape[1]
    if latent_dim == 0:
        return xyz

    batch_vecs = latent_vec.view(latent_vec.shape[0], 1, latent_dim).repeat(1, xyz.shape[0], 1)
    return torch.cat([batch_vecs.reshape(-1, latent_dim), xyz], dim=1)


def _grid_spec(N, grid_bounds=None):
    resolution = np.asarray([N, N, N] if np.isscalar(N) else N, dtype=np.int64)
    if resolution.shape != (3,) or np.any(resolution < 2):
        raise ValueError(f"Grid resolution must contain three values >= 2, got {N}")

    bounds = np.asarray(
        [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        if grid_bounds is None
        else grid_bounds,
        dtype=np.float64,
    )
    if bounds.shape != (6,):
        raise ValueError(f"Grid bounds must be [xmin, ymin, zmin, xmax, ymax, zmax], got {bounds}")
    minimum, maximum = bounds[:3], bounds[3:]
    if not np.isfinite(bounds).all() or np.any(maximum <= minimum):
        raise ValueError(f"Grid bounds must be finite with positive extents, got {bounds}")

    spacing = (maximum - minimum) / (resolution - 1)
    return tuple(int(value) for value in resolution), minimum, maximum, spacing


def _decoder_device(decoder, latent_vec=None, device=None):
    if device is not None:
        return torch.device(device)
    if isinstance(latent_vec, torch.Tensor):
        return latent_vec.device
    try:
        return next(decoder.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _clean_mesh(mesh, validate=False):
    mesh = mesh.process(validate=validate)

    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    elif hasattr(mesh, "remove_duplicate_faces"):
        mesh.remove_duplicate_faces()

    if hasattr(mesh, "nondegenerate_faces"):
        mesh.update_faces(mesh.nondegenerate_faces())
    elif hasattr(mesh, "remove_degenerate_faces"):
        mesh.remove_degenerate_faces()

    mesh.remove_unreferenced_vertices()
    return mesh


def get_udf_normals_grid_slow(
    decoder,
    latent_vec,
    N=56,
    max_batch=int(2 ** 20),
    fourier=False,
    grid_bounds=None,
    device=None,
):
    """
    Fills a dense regular grid by querying the decoder network.
    Inputs: 
        decoder: coordinate network to evaluate
        latent_vec: conditioning vector
        N: scalar grid size or per-axis ``(Nx, Ny, Nz)`` resolution
        max_batch: number of points we can simultaneously evaluate
        fourier: are xyz coordinates encoded with fourier?
        grid_bounds: optional query bounds ``[xmin, ymin, zmin, xmax, ymax, zmax]``
        device: decoder device; inferred from the latent or decoder when omitted
    Returns:
        df_values: (N,N,N) tensor representing distance field values on the grid
        vecs: (N,N,N,3) tensor representing gradients values on the grid, only for locations with a small
                distance field value
        samples: (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
    """
    decoder.eval()
    resolution, grid_min, _, spacing = _grid_spec(N, grid_bounds)
    nx, ny, nz = resolution
    decoder_device = _decoder_device(decoder, latent_vec, device)
    ################
    # 1: setting up the empty grid
    ################
    num_samples = nx * ny * nz
    overall_index = torch.arange(num_samples, dtype=torch.long)
    samples = torch.zeros(num_samples, 7)
    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % nz
    samples[:, 1] = torch.div(overall_index, nz, rounding_mode="floor") % ny
    samples[:, 0] = torch.div(overall_index, ny * nz, rounding_mode="floor")
    # transform first 3 columns
    # to be the x, y, z coordinate
    for axis in range(3):
        samples[:, axis] = samples[:, axis] * float(spacing[axis]) + float(grid_min[axis])
    samples.requires_grad = False
    ################
    # 2: Run forward pass to fill the grid
    ################
    head = 0
    ## FIRST: fill distance field grid without gradients
    while head < num_samples:
        # xyz coords
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].clone().to(decoder_device)
        # Create input
        if fourier:
            xyz = ws.fourier_transform(sample_subset)
        else:
            xyz = sample_subset
        input = _make_decoder_input(latent_vec, xyz)
        # Run forward pass
        with torch.no_grad():
            df = decoder(input)
        # Store df
        samples[head : min(head + max_batch, num_samples), 3] = df.squeeze(-1).detach().cpu()
        # Next iter
        head += max_batch
    #
    ## THEN: compute gradients only where needed,
    # ie. where the predicted df value is small
    max_batch = max_batch // 4
    norm_mask = samples[:, 3] < 2 * float(np.max(spacing))
    norm_idx = torch.where(norm_mask)[0]
    head, num_samples = 0, norm_idx.shape[0]
    while head < num_samples:
        # Find the subset of indices to compute:
        # -> a subset of indices where normal computations are needed
        sample_subset_mask = torch.zeros_like(norm_mask)
        sample_subset_mask[norm_idx[head]: norm_idx[min(head + max_batch, num_samples) - 1] + 1] = True
        sample_subset_mask = norm_mask * sample_subset_mask
        # xyz coords
        sample_subset = samples[sample_subset_mask, 0:3].clone().to(decoder_device)
        sample_subset.requires_grad = True
        # Create input
        if fourier:
            xyz = ws.fourier_transform(sample_subset)
        else:
            xyz = sample_subset
        input = _make_decoder_input(latent_vec, xyz)
        # Run forward pass
        df = decoder(input)
        # Compute and store normalized vectors pointing towards the surface
        grad = torch.autograd.grad(df.sum(), sample_subset, only_inputs=True)[0].detach()
        samples[sample_subset_mask, 4:] = - F.normalize(grad, dim=1).cpu()
        # Next iter
        head += max_batch
    #
    # Separate values in DF / gradients
    df_values = samples[:, 3]
    df_values = df_values.reshape(*resolution)
    vecs = samples[:, 4:]
    vecs = vecs.reshape(*resolution, 3)
    return df_values, vecs, samples


def get_udf_normals_grid_fast(
    decoder,
    latent_vec,
    samples,
    indices,
    N=56,
    max_batch=int(2 ** 18),
    fourier=False,
    grid_bounds=None,
    device=None,
):
    """
    Updates the N*N*N regular grid by querying the decoder network ONLY AT INDICES
    Inputs: 
        decoder: coordinate network to evaluate
        latent_vec: conditioning vector
        samples: already computed (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
                    for a previous latent_vec, which is assumed to be close to the current one.
        indices: tensor representing the coordinates that need updating
        N: scalar grid size or per-axis ``(Nx, Ny, Nz)`` resolution
        max_batch: number of points we can simultaneously evaluate
        fourier: are xyz coordinates encoded with fourier?
        grid_bounds: bounds used to create ``samples``
        device: decoder device; inferred from the latent or decoder when omitted
    Returns:
        df_values: (N,N,N) tensor representing distance field values on the grid
        vecs: (N,N,N,3) tensor representing gradients values on the grid, only for locations with a small
                distance field value
        samples: (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
    """
    resolution, _, _, _ = _grid_spec(N, grid_bounds)
    decoder_device = _decoder_device(decoder, latent_vec, device)
    # If there is no indicesm fallback to the slow version
    if indices is None or samples is None:
        return get_udf_normals_grid_slow(
            decoder,
            latent_vec,
            N,
            max_batch,
            fourier,
            grid_bounds=grid_bounds,
            device=decoder_device,
        )
    decoder.eval()
    ################
    # 1: setting up the empty grid: no longer needed
    ################
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    num_samples = indices.shape[0]
    samples.requires_grad = False
    ################
    # 2: Run forward pass to fill the grid, only where needed
    ################
    head = 0
    ## Fill distance field grid with gradients
    while head < num_samples:
        # xyz coords
        sample_subset = samples[
            indices[head : min(head + max_batch, num_samples)], 0:3
        ].clone().to(decoder_device)
        sample_subset.requires_grad = True
        # Create input
        if fourier:
            xyz = ws.fourier_transform(sample_subset)
        else:
            xyz = sample_subset
        input = _make_decoder_input(latent_vec, xyz)
        # Run forward pass
        df = decoder(input)
        # Store df
        samples[indices[head : min(head + max_batch, num_samples)], 3] = df.squeeze(-1).detach().cpu()
        # Compute and store normalized vectors pointing towards the surface
        grad = torch.autograd.grad(df.sum(), sample_subset, only_inputs=True)[0].detach()
        samples[indices[head : min(head + max_batch, num_samples)], 4:] = - F.normalize(grad, dim=1).cpu()
        # Next iter
        head += max_batch
    # Separate values in DF / gradients
    df_values = samples[:, 3]
    df_values = df_values.reshape(*resolution)
    vecs = samples[:, 4:]
    vecs = vecs.reshape(*resolution, 3)
    return df_values, vecs, samples


def get_udf_normals_grid_streamed(
    decoder,
    latent_vec,
    N,
    max_batch=262144,
    fourier=False,
    grid_bounds=None,
    device=None,
):
    """Fill only the dense UDF/normal arrays; generate XYZ batches on the GPU.

    Unlike the legacy path this does not retain XYZ plus UDF plus normals in an
    ``N**3 x 7`` CPU tensor, and it gathers near-surface gradient points directly
    instead of allocating a full-grid boolean mask for every gradient batch.
    """
    decoder.eval()
    resolution, grid_min, _, spacing = _grid_spec(N, grid_bounds)
    nx, ny, nz = resolution
    num_samples = nx * ny * nz
    decoder_device = _decoder_device(decoder, latent_vec, device)
    grid_min_t = torch.as_tensor(grid_min, device=decoder_device, dtype=torch.float32)
    spacing_t = torch.as_tensor(spacing, device=decoder_device, dtype=torch.float32)
    df_values = torch.empty(num_samples, dtype=torch.float32)

    def coordinates(flat_indices):
        flat_indices = flat_indices.to(decoder_device)
        i = torch.div(flat_indices, ny * nz, rounding_mode="floor")
        remainder = flat_indices - i * ny * nz
        j = torch.div(remainder, nz, rounding_mode="floor")
        k = remainder - j * nz
        return grid_min_t + torch.stack((i, j, k), dim=1).float() * spacing_t

    for start in range(0, num_samples, max_batch):
        stop = min(start + max_batch, num_samples)
        flat = torch.arange(start, stop, device=decoder_device, dtype=torch.long)
        xyz = coordinates(flat)
        encoded = ws.fourier_transform(xyz) if fourier else xyz
        with torch.no_grad():
            values = decoder(_make_decoder_input(latent_vec, encoded))
        df_values[start:stop] = values.reshape(-1).float().cpu()

    normals = torch.zeros((num_samples, 3), dtype=torch.float32)
    near_indices = torch.where(df_values < 2 * float(np.max(spacing)))[0]
    gradient_batch = max(1, max_batch // 4)
    for start in range(0, len(near_indices), gradient_batch):
        indices = near_indices[start : start + gradient_batch]
        xyz = coordinates(indices).requires_grad_(True)
        encoded = ws.fourier_transform(xyz) if fourier else xyz
        values = decoder(_make_decoder_input(latent_vec, encoded))
        gradients = torch.autograd.grad(values.sum(), xyz, only_inputs=True)[0]
        normals[indices] = (-F.normalize(gradients, dim=1)).detach().cpu()
    return df_values.reshape(*resolution), normals.reshape(*resolution, 3)


def get_mesh_udf_fast(decoder, latent_vec, samples=None, indices=None, N_MC=128, fourier=False,
                        gradient=True, eps=0.1, border_gradients=False, smooth_borders=False,
                        vertex_max_batch=int(2 ** 16), surface_threshold=None,
                        grid_bounds=None, device=None, streamed_grid=False,
                        query_batch=int(2 ** 18), fast_cleanup=False,
                        mc_tile_cells=0):
    """
    Computes a triangulated mesh from a distance field network conditioned on the latent vector
    Inputs: 
        decoder: coordinate network to evaluate
        latent_vec: conditioning vector
        samples: already computed (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
                    for a previous latent_vec, which is assumed to be close to the current one, if any
        indices: tensor representing the coordinates that need updating in the previous samples tensor (to speed
                    up iterations)
        N_MC: scalar grid size or per-axis ``(Nx, Ny, Nz)`` resolution
        fourier: are xyz coordinates encoded with fourier?
        gradient: do we need gradients?
        eps: length of the normal vectors used to derive gradients
        border_gradients: add a special case for border gradients?
        smooth_borders: do we smooth borders with a Laplacian?
        vertex_max_batch: maximum number of marching-cubes vertices to evaluate at once
        surface_threshold: maximum predicted UDF at every face vertex; defaults to voxel_size / 6
        grid_bounds: optional query bounds ``[xmin, ymin, zmin, xmax, ymax, zmax]``
        device: decoder device; inferred from the latent or decoder when omitted
    Returns:
        verts: vertices of the mesh
        faces: faces of the mesh
        mesh: trimesh object of the mesh
        samples: (N**3, 7) tensor representing (x,y,z, distance field, grad_x, grad_y, grad_z)
        indices: tensor representing the coordinates that need updating in the next iteration
    """
    resolution, grid_min, grid_max, spacing = _grid_spec(N_MC, grid_bounds)
    decoder_device = _decoder_device(decoder, latent_vec, device)
    ### 1: sample grid
    if streamed_grid and samples is None and indices is None:
        df_values, normals = get_udf_normals_grid_streamed(
            decoder,
            latent_vec,
            resolution,
            max_batch=query_batch,
            fourier=fourier,
            grid_bounds=np.concatenate([grid_min, grid_max]),
            device=decoder_device,
        )
        samples = None
    else:
        df_values, normals, samples = get_udf_normals_grid_fast(
            decoder,
            latent_vec,
            samples,
            indices,
            N=resolution,
            max_batch=query_batch,
            fourier=fourier,
            grid_bounds=np.concatenate([grid_min, grid_max]),
            device=decoder_device,
        )
    df_values[df_values < 0] = 0
    ### 2: run our custom MC on it
    df_numpy = df_values.cpu().detach().numpy()
    normals_numpy = normals.cpu().detach().numpy()
    if mc_tile_cells and mc_tile_cells > 0:
        tile_vertices, tile_faces, vertex_offset = [], [], 0
        cell_counts = np.asarray(resolution) - 1
        for sx in range(0, int(cell_counts[0]), mc_tile_cells):
            ex = min(sx + mc_tile_cells, int(cell_counts[0]))
            for sy in range(0, int(cell_counts[1]), mc_tile_cells):
                ey = min(sy + mc_tile_cells, int(cell_counts[1]))
                for sz in range(0, int(cell_counts[2]), mc_tile_cells):
                    ez = min(sz + mc_tile_cells, int(cell_counts[2]))
                    tile_normals = normals_numpy[sx : ex + 1, sy : ey + 1, sz : ez + 1]
                    if not np.any(tile_normals):
                        continue
                    try:
                        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
                            tile_v, tile_f, _, _ = udf_mc_lewiner(
                                df_numpy[sx : ex + 1, sy : ey + 1, sz : ez + 1],
                                tile_normals,
                                spacing=spacing,
                            )
                    except RuntimeError as exc:
                        if "No surface found" in str(exc):
                            continue
                        raise
                    tile_v += grid_min + np.asarray((sx, sy, sz)) * spacing
                    tile_vertices.append(tile_v)
                    tile_faces.append(tile_f + vertex_offset)
                    vertex_offset += len(tile_v)
        if not tile_vertices:
            return None, None, None
        verts = np.concatenate(tile_vertices)
        faces = np.concatenate(tile_faces)
    else:
        verts, faces, _, _ = udf_mc_lewiner(df_numpy, normals_numpy, spacing=spacing)
        verts = verts + grid_min
    ### 3: evaluate vertices DF, and remove the ones that are too far
    if vertex_max_batch < 1:
        raise ValueError("vertex_max_batch must be at least 1")

    pred_df_chunks = []
    with torch.no_grad():
        for start in range(0, len(verts), vertex_max_batch):
            verts_torch = torch.from_numpy(
                verts[start : start + vertex_max_batch]
            ).float().to(decoder_device)
            if fourier:
                xyz = ws.fourier_transform(verts_torch)
            else:
                xyz = verts_torch
            pred_df_chunks.append(
                decoder(_make_decoder_input(latent_vec, xyz)).detach().cpu()
            )
    pred_df_verts = torch.cat(pred_df_chunks, dim=0).numpy()
    # An absolute override avoids an overly strict voxel-relative filter at high resolutions.
    if surface_threshold is None:
        surface_threshold = float(np.max(spacing)) / 6
    elif surface_threshold < 0:
        raise ValueError("surface_threshold must be non-negative")
    filtered_faces = faces[
        np.max(pred_df_verts[faces], axis=1)[:, 0] < surface_threshold
    ]
    filtered_mesh = trimesh.Trimesh(verts, filtered_faces)
    ### 4: clean the mesh a bit
    # Remove NaNs, flat triangles, duplicate faces
    filtered_mesh = _clean_mesh(filtered_mesh, validate=False)
    # Fill single triangle holes
    filtered_mesh.fill_holes()

    if fast_cleanup:
        # One post-fill pass removes any duplicate/degenerate faces introduced
        # while closing single-triangle holes without the legacy 10-pass loop.
        filtered_mesh = _clean_mesh(filtered_mesh, validate=False)
    else:
        filtered_mesh_2 = trimesh.Trimesh(filtered_mesh.vertices, filtered_mesh.faces)
        # Re-process the mesh until it is stable:
        n_verts, n_faces, n_iter = 0, 0, 0
        while (n_verts, n_faces) != (len(filtered_mesh_2.vertices), len(filtered_mesh_2.faces)) and n_iter<10:
            filtered_mesh_2 = _clean_mesh(filtered_mesh_2, validate=False)
            (n_verts, n_faces) = (len(filtered_mesh_2.vertices), len(filtered_mesh_2.faces))
            n_iter += 1
            filtered_mesh_2 = trimesh.Trimesh(filtered_mesh_2.vertices, filtered_mesh_2.faces)
        filtered_mesh = trimesh.Trimesh(filtered_mesh_2.vertices, filtered_mesh_2.faces)

    if smooth_borders:
        # Identify borders: those appearing only once
        border_edges = trimesh.grouping.group_rows(filtered_mesh.edges_sorted, require_count=1)

        # Build a dictionnary of (u,l): l is the list of vertices that are adjacent to u
        neighbours  = defaultdict(lambda: [])
        for (u,v) in filtered_mesh.edges_sorted[border_edges]:
            neighbours[u].append(v)
            neighbours[v].append(u)
        border_vertices = np.array(list(neighbours.keys()))

        # Build a sparse matrix for computing laplacian
        pos_i, pos_j = [], []
        for k, ns in enumerate(neighbours.values()):
            for j in ns:
                pos_i.append(k)
                pos_j.append(j)

        sparse = coo_matrix((np.ones(len(pos_i)),   # put ones
                            (pos_i, pos_j)),        # at these locations
                            shape=(len(border_vertices), len(filtered_mesh.vertices)))

        # Smoothing operation:
        lambda_ = 0.3
        for _ in range(5):
            border_neighbouring_averages = sparse @ filtered_mesh.vertices / sparse.sum(axis=1)
            laplacian = border_neighbouring_averages - filtered_mesh.vertices[border_vertices]
            filtered_mesh.vertices[border_vertices] = filtered_mesh.vertices[border_vertices] + lambda_ * laplacian

    if not gradient:
        return (
            torch.as_tensor(filtered_mesh.vertices, dtype=torch.float32, device=decoder_device),
            torch.as_tensor(filtered_mesh.faces, dtype=torch.long, device=decoder_device),
            filtered_mesh,
        )
    else:
        ### 5: use the mesh to compute normals
        normals = trimesh.geometry.weighted_vertex_normals( vertex_count=len(filtered_mesh.vertices),
                                                            faces=filtered_mesh.faces,
                                                            face_normals=filtered_mesh.face_normals,
                                                            face_angles=filtered_mesh.face_angles)
        ### 6: evaluate the DF around each vertex, based on normals
        normals = torch.as_tensor(normals, dtype=torch.float32, device=decoder_device)
        verts = torch.as_tensor(filtered_mesh.vertices, dtype=torch.float32, device=decoder_device)
        if fourier:
            xyz_s1 = ws.fourier_transform(verts + eps * normals)
            xyz_s2 = ws.fourier_transform(verts - eps * normals)
        else:
            xyz_s1 = verts + eps * normals
            xyz_s2 = verts - eps * normals
        s1 = decoder(_make_decoder_input(latent_vec, xyz_s1))
        s2 = decoder(_make_decoder_input(latent_vec, xyz_s2))
        # Re-plug differentiability here, by this rewriting trick
        new_verts = verts - eps * s1 * normals + eps * s2 * normals

        ## Compute indices needed for re-evaluation at the next iteration
        # fetch bins that are activated
        vertex_indices = np.floor(
            (new_verts.detach().cpu().numpy() - grid_min) / spacing
        ).astype(int)
        vertex_indices = np.clip(vertex_indices, 0, np.asarray(resolution) - 1)
        i, j, k = vertex_indices.T
        nx, ny, nz = resolution
        # find points around
        next_samples = i * ny * nz + j * nz + k
        next_samples_ip = np.minimum(i + 1, nx - 1) * ny * nz + j * nz + k
        next_samples_jp = i * ny * nz + np.minimum(j + 1, ny - 1) * nz + k
        next_samples_kp = i * ny * nz + j * nz + np.minimum(k + 1, nz - 1)
        next_samples_im = np.maximum(i - 1, 0) * ny * nz + j * nz + k
        next_samples_jm = i * ny * nz + np.maximum(j - 1, 0) * nz + k
        next_samples_km = i * ny * nz + j * nz + np.maximum(k - 1, 0)
        # Concatenate
        next_indices = np.concatenate((next_samples, next_samples_ip, next_samples_jp,
                        next_samples_kp, next_samples_im, next_samples_jm, next_samples_km))

        if border_gradients:
            ### 7: Add gradients at the surface borders?
            # Identify borders
            border_edges = trimesh.grouping.group_rows(filtered_mesh.edges_sorted, require_count=1)

            # Build a dictionnary of (u,v) edges, such that each vertex on the border
            # gets associated to exactly one border edge
            border_edges_dict = {}
            for (u,v) in filtered_mesh.edges_sorted[border_edges]:
                border_edges_dict[u] = v
                border_edges_dict[v] = u
            u_v_border = np.array(list(border_edges_dict.items()))
            u_border = u_v_border[:,0]    # split border edges (u,v) into u and v arrays
            v_border = u_v_border[:,1]

            # For each vertex on the border, take the cross product between 
            # its normal and the border's edge
            normals_border = normals[u_border]
            edge_border = filtered_mesh.vertices[v_border] - filtered_mesh.vertices[u_border]
            edge_border = torch.as_tensor(edge_border, dtype=torch.float32, device=decoder_device)
            out_vec = torch.cross(edge_border, normals_border, dim=1)
            out_vec = out_vec / (torch.norm(out_vec, dim=1, keepdim=True) + 1e-6) # make it unit length

            # Then we need to orient the out_vec such that they point outwards
            # To do so, we evaluate at +- their offset, and take the corresponding max DF value
            border_verts = torch.as_tensor(
                filtered_mesh.vertices[u_border],
                dtype=torch.float32,
                device=decoder_device,
            )
            if fourier:
                xyz_s1_border = ws.fourier_transform(border_verts + 3*eps * out_vec)
                xyz_s2_border = ws.fourier_transform(border_verts - 3*eps * out_vec)
            else:
                xyz_s1_border = border_verts + 3*eps * out_vec
                xyz_s2_border = border_verts - 3*eps * out_vec

            s1_border = decoder(_make_decoder_input(latent_vec, xyz_s1_border))
            s2_border = decoder(_make_decoder_input(latent_vec, xyz_s2_border))
            s1s2 = torch.stack((s1_border,s2_border))
            sign_out_vec = -torch.argmax(s1s2, dim=0)*2 + 1
            out_vec = sign_out_vec * out_vec

            # Filter out the verts borders for which a displacement of out_vec
            # still evaluates at < eps DF, ie. verts classified as borders which are not really so
            u_border_filtered = u_border[((s1_border+s2_border)[:,0] > eps).detach().cpu().numpy()]
            out_vec_filtered = out_vec[(s1_border+s2_border)[:,0] > eps]
            out_df_filtered = torch.max(s1_border, s2_border)[(s1_border+s2_border) > eps]

            # Plug gradients to verts positions
            s_border = (eps * (out_df_filtered - out_df_filtered.detach())).unsqueeze(-1)   # Fake zero, just to pass grads
            new_verts[u_border_filtered] = new_verts[u_border_filtered] - s_border * out_vec_filtered

        return (
            new_verts,
            torch.as_tensor(filtered_mesh.faces, dtype=torch.long, device=decoder_device),
            filtered_mesh,
            samples,
            next_indices,
        )


def _query_decoder(decoder, latent_vec, points, max_batch, fourier, device, gradients=False):
    """Query UDF values, and optionally normalized negative gradients, in batches."""
    values, vectors = [], []
    for start in range(0, len(points), max_batch):
        xyz = points[start : start + max_batch].to(device)
        xyz.requires_grad_(gradients)
        encoded = ws.fourier_transform(xyz) if fourier else xyz
        if gradients:
            value = decoder(_make_decoder_input(latent_vec, encoded))
            grad = torch.autograd.grad(value.sum(), xyz, only_inputs=True)[0]
            vectors.append((-F.normalize(grad, dim=1)).detach().cpu())
        else:
            with torch.no_grad():
                value = decoder(_make_decoder_input(latent_vec, encoded))
        values.append(value.detach().cpu())
    value = torch.cat(values, dim=0).reshape(-1)
    vector = torch.cat(vectors, dim=0) if gradients else None
    return value, vector


def get_mesh_udf_bricked(
    decoder,
    latent_vec,
    N_MC=1024,
    *,
    fourier=False,
    brick_cells=64,
    cull_margin=1.25,
    query_batch=262144,
    vertex_max_batch=262144,
    surface_threshold=None,
    grid_bounds=None,
    device=None,
):
    """Sparse narrow-band MeshUDF extraction using independently meshed bricks.

    A coarse lattice with half-brick spacing conservatively identifies bricks
    which may contain the surface.  Active bricks share their boundary grid
    planes, and the final mesh welds coincident vertices once globally.
    """
    resolution, grid_min, grid_max, spacing = _grid_spec(N_MC, grid_bounds)
    if brick_cells < 4 or query_batch < 1 or vertex_max_batch < 1:
        raise ValueError("brick_cells must be >= 4 and batch sizes must be positive")
    if not np.isfinite(cull_margin) or cull_margin <= 0:
        raise ValueError("cull_margin must be positive and finite")
    decoder_device = _decoder_device(decoder, latent_vec, device)
    decoder.eval()

    # Sample a shared lattice every half brick.  Any point in a brick lies no
    # farther than sqrt(3) * half_step / 2 from one of these samples.
    cell_counts = np.asarray(resolution, dtype=np.int64) - 1
    coarse_step = max(2, brick_cells // 2)
    coarse_axes = []
    for count in cell_counts:
        axis = np.arange(0, count + 1, coarse_step, dtype=np.int64)
        if axis[-1] != count:
            axis = np.append(axis, count)
        coarse_axes.append(axis)
    coarse_ijk = torch.cartesian_prod(
        *(torch.as_tensor(axis, dtype=torch.float32, device=decoder_device) for axis in coarse_axes)
    )
    coarse_xyz = torch.as_tensor(grid_min, device=decoder_device, dtype=torch.float32) + coarse_ijk * torch.as_tensor(
        spacing, device=decoder_device, dtype=torch.float32
    )
    coarse_values, _ = _query_decoder(
        decoder, latent_vec, coarse_xyz, query_batch, fourier, decoder_device
    )
    coarse_values = coarse_values.reshape(*(len(axis) for axis in coarse_axes)).numpy()

    active = []
    brick_starts = [np.arange(0, count, brick_cells, dtype=np.int64) for count in cell_counts]
    cover_radius = cull_margin * np.sqrt(3.0) * coarse_step * float(np.max(spacing)) * 0.5
    for sx in brick_starts[0]:
        ix = np.where((coarse_axes[0] >= sx) & (coarse_axes[0] <= min(sx + brick_cells, cell_counts[0])))[0]
        for sy in brick_starts[1]:
            iy = np.where((coarse_axes[1] >= sy) & (coarse_axes[1] <= min(sy + brick_cells, cell_counts[1])))[0]
            for sz in brick_starts[2]:
                iz = np.where((coarse_axes[2] >= sz) & (coarse_axes[2] <= min(sz + brick_cells, cell_counts[2])))[0]
                if coarse_values[np.ix_(ix, iy, iz)].min() <= cover_radius:
                    active.append((int(sx), int(sy), int(sz)))

    all_vertices, all_faces = [], []
    vertex_offset = 0
    spacing_t = torch.as_tensor(spacing, device=decoder_device, dtype=torch.float32)
    grid_min_t = torch.as_tensor(grid_min, device=decoder_device, dtype=torch.float32)
    normal_band = 2.0 * float(np.max(spacing))
    for sx, sy, sz in tqdm.tqdm(active, desc="MeshUDF bricks", leave=False):
        ends = np.minimum(np.asarray((sx, sy, sz)) + brick_cells, cell_counts)
        shape = tuple((ends - np.asarray((sx, sy, sz)) + 1).tolist())
        axes = [torch.arange(n, device=decoder_device, dtype=torch.float32) for n in shape]
        local_ijk = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)
        global_ijk = local_ijk + torch.tensor((sx, sy, sz), device=decoder_device)
        xyz = grid_min_t + global_ijk * spacing_t
        values, _ = _query_decoder(
            decoder, latent_vec, xyz, query_batch, fourier, decoder_device
        )
        values.clamp_min_(0)
        normals = torch.zeros((len(xyz), 3), dtype=torch.float32)
        near = values < normal_band
        if near.any():
            _, near_normals = _query_decoder(
                decoder,
                latent_vec,
                xyz[near.to(xyz.device)],
                max(1, query_batch // 4),
                fourier,
                decoder_device,
                gradients=True,
            )
            normals[near] = near_normals
        try:
            # The bundled Cython kernel prints one diagnostic per ambiguous
            # zero-gradient case; suppress it for brick-wise extraction.
            with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
                verts, faces, _, _ = udf_mc_lewiner(
                    values.reshape(shape).numpy(), normals.reshape(*shape, 3).numpy(), spacing=spacing
                )
        except RuntimeError as exc:
            if "No surface found" in str(exc):
                continue
            raise
        if not len(verts) or not len(faces):
            continue
        verts += grid_min + np.asarray((sx, sy, sz)) * spacing

        pred_values = []
        for start in range(0, len(verts), vertex_max_batch):
            verts_t = torch.as_tensor(
                verts[start : start + vertex_max_batch], device=decoder_device, dtype=torch.float32
            )
            value, _ = _query_decoder(
                decoder, latent_vec, verts_t, vertex_max_batch, fourier, decoder_device
            )
            pred_values.append(value.numpy())
        pred_values = np.concatenate(pred_values)
        threshold = float(np.max(spacing)) / 6 if surface_threshold is None else float(surface_threshold)
        faces = faces[np.max(pred_values[faces], axis=1) < threshold]
        if not len(faces):
            continue
        all_vertices.append(verts)
        all_faces.append(faces + vertex_offset)
        vertex_offset += len(verts)

    if not all_vertices:
        return None, None, None
    mesh = trimesh.Trimesh(
        vertices=np.concatenate(all_vertices), faces=np.concatenate(all_faces), process=False
    )
    # One global weld is sufficient; avoid the dense path's repeated whole-mesh processing.
    mesh.merge_vertices(digits_vertex=7)
    mesh = _clean_mesh(mesh, validate=False)
    mesh.fill_holes()
    return (
        torch.as_tensor(mesh.vertices, dtype=torch.float32, device=decoder_device),
        torch.as_tensor(mesh.faces, dtype=torch.long, device=decoder_device),
        mesh,
    )


def optimize(decoder, iterations, latent_start, mesh_target, lr=5e-3, fourier=False,
            MC=128, border_gradients=False, smooth_borders=False, out_dir=''):
    ### Create init latent code and its optimizer
    latent = latent_start.unsqueeze(0).detach().clone().cuda()
    latent.requires_grad = True
    optimizer = torch.optim.Adam([latent], lr=lr)

    # Turn trimesh mesh into pytorch3D
    target_mesh_torch = Meshes(torch.tensor(mesh_target.vertices).float().unsqueeze(0).cuda(),
                                torch.tensor(mesh_target.faces).long().unsqueeze(0).cuda())
    
    samples, indices = None, None
    for e in tqdm.tqdm(range(iterations+1)):
        decoder.eval()
        # samples and indices store the values and locations of the field for a lazy
        # re-evaluation strategy during the optimization. For stability, it is reset for the first
        # 20 steps and every 25 steps
        if e%25 == 0 or e < 20:
            samples, indices = None, None
        
        # Forward pass: get current mesh
        optimizer.zero_grad()
        pred_v, pred_f, pred_mesh, samples, indices = get_mesh_udf_fast(decoder, latent, samples, indices,
                                                                        eps=0.01, N_MC=MC, gradient=True, fourier=fourier,
                                                                        border_gradients=border_gradients, smooth_borders=smooth_borders)
        optimizer.zero_grad() # because we computed analytical normals in get_mesh_udf_fast(), we need to zero-out gradients now
        
        # Sample points on its surface
        pred_mesh_torch = Meshes(pred_v.unsqueeze(0), pred_f.unsqueeze(0))
        pred_mesh_pts = sample_points_from_meshes(pred_mesh_torch, num_samples=10000)
        
        # Chamfer loss wrt. target data points
        target_mesh_pts = sample_points_from_meshes(target_mesh_torch, num_samples=10000)
        loss, _ = chamfer_distance(target_mesh_pts, pred_mesh_pts)
        # print((loss * 10000).item())

        # Backwards pass + update
        loss.backward()
        optimizer.step()

        # Save pred mesh
        if (e < 100 and e % 5 == 0) or e % 50 == 0:
            _ = pred_mesh.export(os.path.join(out_dir, f'A_{e:04d}.ply'))
    
    return latent, pred_mesh


def main_function(experiment_directory, continue_from, out_dir, MC, iterations,
                    shape_a, shape_b, border_gradients, smooth_borders):
    specs = ws.load_experiment_specifications(experiment_directory)
    arch = __import__("lib.models." + specs["NetworkDecoder"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]

    #########################################
    ## UDF decoder
    #########################################
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"]).cuda()
    decoder = torch.nn.DataParallel(decoder)

    #########################################
    ## Reload model weights + latents vectors
    #########################################
    print('Using weights and codes from epoch "{}"'.format(continue_from))
    lat_vecs = ws.load_latent_vectors(experiment_directory, continue_from + ".pth")
    _ = ws.load_model_parameters_decoder(experiment_directory, continue_from, decoder)
    decoder.eval()

    print("Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())))
    print("Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.shape[0] * lat_vecs.shape[1], lat_vecs.shape[0], lat_vecs.shape[1]))

    #########################################
    ## Reconstruct and optimize
    #########################################
    
    ## Reconstruct source (=A) and target (=B) meshes
    # Source
    latent_start = lat_vecs[shape_a].detach()
    _, _, mesh_start = get_mesh_udf_fast(decoder, latent_start.unsqueeze(0), N_MC=MC, gradient=False,
                                        smooth_borders=smooth_borders, fourier=specs["NetworkSpecs"]["fourier"])
    _ = mesh_start.export(os.path.join(out_dir, 'A.ply'))
    # Target
    latent_target = lat_vecs[shape_b].detach()
    _, _, mesh_target = get_mesh_udf_fast(decoder, latent_target.unsqueeze(0), N_MC=MC, gradient=False,
                                        smooth_borders=smooth_borders, fourier=specs["NetworkSpecs"]["fourier"])
    _ = mesh_target.export(os.path.join(out_dir, 'B.ply'))

    ## Launch optimization from A to B with CHD loss
    optim_code, optim_mesh = optimize(
                    decoder,
                    iterations,
                    latent_start,
                    mesh_target,
                    lr=5e-3,
                    fourier=specs["NetworkSpecs"]["fourier"],
                    MC=MC,
                    border_gradients=border_gradients,
                    smooth_borders=smooth_borders,
                    out_dir=out_dir
                )
                
    # Save optimized code and corresponding mesh
    _ = optim_mesh.export(os.path.join(out_dir, 'A_optimized.ply'))
    torch.save(optim_code.unsqueeze(0), os.path.join(out_dir, 'A_optimized.pth'))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Reconstruct training shapes with a trained DeepSDF autodecoder (+latents)")
    arg_parser.add_argument(
        "--experiment", "-e", dest="experiment_directory", required=True,
        help="The experiment directory. This directory should include "
        + "experiment specifications in 'specs.json', and logging will be "
        + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--continue", "-c", dest="continue_from", default="latest",
        help="A snapshot to continue from. This can be 'latest' to continue"
        + "from the latest running snapshot, or an integer corresponding to "
        + "an epochal snapshot.",
    )
    arg_parser.add_argument(
        "--MC", type=int, default=256,
        help="What is marching cubes resolution?",
    )
    arg_parser.add_argument(
        "--A", type=int, default=0,
        help="Starting shape id?",
    )
    arg_parser.add_argument(
        "--B", type=int, default=2,
        help="Target shape id?",
    )
    arg_parser.add_argument(
        "--no_border_gradients", dest="no_border_gradients", action="store_true",
        help="Do we skip the special case for gradients on the borders (to allow surface extension/shrinking?)",
    )
    arg_parser.add_argument(
        "--no_smooth_borders", dest="no_smooth_borders", action="store_true",
        help="Do we skip the border smoothing step?",
    )
    arg_parser.add_argument(
        "--iters", type=int, default=800,
        help="Optimization steps",
    )

    args = arg_parser.parse_args()

    out_dir = os.path.join(args.experiment_directory, f'optim_{args.A}_to_{args.B}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print('======================================')
    print(f'Meshing the UDF directly')
    print(f'And optimizing CHD loss from shape {args.A} to shape {args.B}')
    print(f'MC resolution: {args.MC}')
    print(f'Results will be stored in: {out_dir}')
    print('======================================')

    main_function(  args.experiment_directory, args.continue_from, out_dir,  args.MC, args.iters,
                    args.A, args.B, not(args.no_border_gradients), not(args.no_smooth_borders))
