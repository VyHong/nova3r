import torch
import torch.nn.functional as F


class SharpEdgeSurfaceLoader:
    def __init__(self, num_uniform_points=4096, num_sharp_points=4096, scale=0.9999, sharpedge_flag=True):
        """
        Args:
            num_uniform_points (int): Points to sample uniformly across the surface.
            num_sharp_points (int): Points to sample specifically along sharp edges.
            scale (float): Bounding box scaling factor.
            sharpedge_flag (bool): If True, appends a 7th dimension indicating point source
                                   (0 for uniform, 1 for sharp edge).
        """
        self.num_uniform_points = num_uniform_points
        self.num_sharp_points = num_sharp_points
        self.scale = scale
        self.sharpedge_flag = sharpedge_flag

    def normalize_mesh_batched(self, vertices, point_masks):
        v_valid_min = vertices.masked_fill(~point_masks.unsqueeze(-1), float("inf"))
        v_valid_max = vertices.masked_fill(~point_masks.unsqueeze(-1), float("-inf"))

        bbox_min = v_valid_min.amin(dim=1, keepdim=True)
        bbox_max = v_valid_max.amax(dim=1, keepdim=True)

        center = (bbox_max + bbox_min) / 2.0
        scale_ = (bbox_max - bbox_min).amax(dim=-1, keepdim=True)
        scale_ = torch.clamp(scale_, min=1e-8)

        normalized_vertices = (vertices - center) * (2.0 * self.scale / scale_)
        return normalized_vertices.masked_fill(~point_masks.unsqueeze(-1), 0.0)

    def compute_normals_and_areas(self, vertices, faces, face_masks):
        B, _, _ = vertices.shape

        # Gather face vertices
        faces_exp = faces.unsqueeze(-1).expand(-1, -1, -1, 3)
        v0 = torch.gather(vertices, 1, faces_exp[:, :, 0, :])
        v1 = torch.gather(vertices, 1, faces_exp[:, :, 1, :])
        v2 = torch.gather(vertices, 1, faces_exp[:, :, 2, :])

        # Face areas and normals
        cross = torch.cross(v1 - v0, v2 - v0, dim=-1)
        face_areas = 0.5 * torch.norm(cross, dim=-1)
        face_normals = F.normalize(cross, p=2, dim=-1)

        if face_masks is not None:
            face_areas = face_areas * face_masks.float()
            face_normals = face_normals * face_masks.unsqueeze(-1).float()

        return v0, v1, v2, face_areas, face_normals

    def sample_surface_batched(self, v0, v1, v2, face_areas, face_normals, num_samples):
        B = face_areas.shape[0]

        probs = face_areas / (face_areas.sum(dim=-1, keepdim=True) + 1e-8)
        sampled_face_idx = torch.multinomial(probs, num_samples, replacement=True)

        idx_exp = sampled_face_idx.unsqueeze(-1).expand(-1, -1, 3)
        s_v0 = torch.gather(v0, 1, idx_exp)
        s_v1 = torch.gather(v1, 1, idx_exp)
        s_v2 = torch.gather(v2, 1, idx_exp)
        sampled_normals = torch.gather(face_normals, 1, idx_exp)

        r1 = torch.rand(B, num_samples, 1, device=v0.device)
        r2 = torch.rand(B, num_samples, 1, device=v0.device)

        sqrt_r1 = torch.sqrt(r1)
        u, v, w = 1.0 - sqrt_r1, sqrt_r1 * (1.0 - r2), sqrt_r1 * r2

        sampled_points = u * s_v0 + v * s_v1 + w * s_v2
        return sampled_points, sampled_normals

    def sample_sharp_edges_batched(self, vertices, faces, v0, v1, v2, face_areas, face_normals, num_samples):
        B, V, _ = vertices.shape
        _, F_num, _ = faces.shape
        device = vertices.device

        # 1. Compute vertex normals (Area-weighted scatter_add)
        vertex_normals = torch.zeros((B, V, 3), device=device)
        weighted_normals = face_normals * face_areas.unsqueeze(-1)
        for i in range(3):
            vertex_normals.scatter_add_(1, faces[:, :, i : i + 1].expand(-1, -1, 3), weighted_normals)
        vertex_normals = F.normalize(vertex_normals, p=2, dim=-1)

        # 2. Compute minimum dot product between vertex normal and connected face normals
        v0_n = torch.gather(vertex_normals, 1, faces[:, :, 0:1].expand(-1, -1, 3))
        v1_n = torch.gather(vertex_normals, 1, faces[:, :, 1:2].expand(-1, -1, 3))
        v2_n = torch.gather(vertex_normals, 1, faces[:, :, 2:3].expand(-1, -1, 3))

        dot0 = (face_normals * v0_n).sum(dim=-1)
        dot1 = (face_normals * v1_n).sum(dim=-1)
        dot2 = (face_normals * v2_n).sum(dim=-1)

        # Note: scatter_reduce requires PyTorch >= 1.12
        min_dot = torch.ones((B, V), device=device)
        min_dot.scatter_reduce_(1, faces[:, :, 0], dot0, reduce="amin", include_self=False)
        min_dot.scatter_reduce_(1, faces[:, :, 1], dot1, reduce="amin", include_self=False)
        min_dot.scatter_reduce_(1, faces[:, :, 2], dot2, reduce="amin", include_self=False)

        # 3. Identify sharp vertices and valid edges
        sharp_mask = min_dot < 0.985  # (B, V)

        # Check if both vertices of an edge are sharp (Edges: v0-v1, v1-v2, v2-v0)
        s0 = torch.gather(sharp_mask, 1, faces[:, :, 0]) & torch.gather(sharp_mask, 1, faces[:, :, 1])
        s1 = torch.gather(sharp_mask, 1, faces[:, :, 1]) & torch.gather(sharp_mask, 1, faces[:, :, 2])
        s2 = torch.gather(sharp_mask, 1, faces[:, :, 2]) & torch.gather(sharp_mask, 1, faces[:, :, 0])
        edge_sharp_mask = torch.stack([s0, s1, s2], dim=-1)  # (B, F, 3)

        # 4. Edge lengths for probability weighting
        edge_lengths = torch.stack([torch.norm(v1 - v0, dim=-1), torch.norm(v2 - v1, dim=-1), torch.norm(v0 - v2, dim=-1)], dim=-1)

        edge_weights = (edge_lengths * edge_sharp_mask.float()).view(B, -1)  # Flatten to (B, F*3)

        # Handle meshes with zero sharp edges to prevent multinomial crash
        is_empty = edge_weights.sum(dim=-1) < 1e-6
        edge_weights[is_empty, :] = 1.0  # Uniform fallback

        # Sample edges
        prob = edge_weights / edge_weights.sum(dim=-1, keepdim=True)
        sampled_idx = torch.multinomial(prob, num_samples, replacement=True)

        # 5. Extract endpoints of sampled edges and interpolate
        idx_exp = sampled_idx.unsqueeze(-1).expand(-1, -1, 3)

        e_v_a = torch.stack([v0, v1, v2], dim=2).view(B, F_num * 3, 3)
        e_v_b = torch.stack([v1, v2, v0], dim=2).view(B, F_num * 3, 3)
        e_n_a = torch.stack([v0_n, v1_n, v2_n], dim=2).view(B, F_num * 3, 3)
        e_n_b = torch.stack([v1_n, v2_n, v0_n], dim=2).view(B, F_num * 3, 3)

        s_v_a = torch.gather(e_v_a, 1, idx_exp)
        s_v_b = torch.gather(e_v_b, 1, idx_exp)
        s_n_a = torch.gather(e_n_a, 1, idx_exp)
        s_n_b = torch.gather(e_n_b, 1, idx_exp)

        # Random linear interpolation along the edge segment
        w = torch.rand(B, num_samples, 1, device=device)
        sampled_points = w * s_v_a + (1.0 - w) * s_v_b
        sampled_normals = w * s_n_a + (1.0 - w) * s_n_b

        return sampled_points, F.normalize(sampled_normals, p=2, dim=-1)

    def __call__(self, batch_data: dict):
        vertices = batch_data["cam_points"].float()
        faces = batch_data["cam_faces"].long()
        device = vertices.device
        B, V, _ = vertices.shape
        _, F_num, _ = faces.shape

        point_masks = batch_data.get("point_masks", torch.ones((B, V), dtype=torch.bool, device=device))
        face_masks = batch_data.get("face_masks", torch.ones((B, F_num), dtype=torch.bool, device=device))

        # 1. Normalize
        norm_vertices = self.normalize_mesh_batched(vertices, point_masks)

        # 2. Precompute geometric properties
        v0, v1, v2, f_areas, f_normals = self.compute_normals_and_areas(norm_vertices, faces, face_masks)

        surfaces_to_cat = []

        # 3. Sample standard uniform surface (Conditional)
        if self.num_uniform_points > 0:
            u_pts, u_nrms = self.sample_surface_batched(v0, v1, v2, f_areas, f_normals, self.num_uniform_points)
            u_data = torch.cat([u_pts, u_nrms], dim=-1)  # (B, N_u, 6)
            if self.sharpedge_flag:
                u_labels = torch.zeros((B, self.num_uniform_points, 1), device=device)
                u_data = torch.cat([u_data, u_labels], dim=-1)  # (B, N_u, 7)
            surfaces_to_cat.append(u_data)

        # 4. Sample sharp edges (Conditional)
        if self.num_sharp_points > 0:
            s_pts, s_nrms = self.sample_sharp_edges_batched(norm_vertices, faces, v0, v1, v2, f_areas, f_normals, self.num_sharp_points)
            s_data = torch.cat([s_pts, s_nrms], dim=-1)  # (B, N_s, 6)
            if self.sharpedge_flag:
                s_labels = torch.ones((B, self.num_sharp_points, 1), device=device)
                s_data = torch.cat([s_data, s_labels], dim=-1)  # (B, N_s, 7)
            surfaces_to_cat.append(s_data)

        # 5. Combine into single tensor
        if len(surfaces_to_cat) == 0:
            raise ValueError("Both num_uniform_points and num_sharp_points are 0. Cannot return empty tensor.")

        combined_surface = torch.cat(surfaces_to_cat, dim=1)
        return combined_surface
