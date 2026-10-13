import argparse
from ast import arg
import os
import sys
import json
import numpy as np
import trimesh
from tqdm import tqdm
from scipy.interpolate import CubicHermiteSpline, interp1d
from scipy.ndimage import gaussian_filter1d
from collections import Counter
from contextlib import contextmanager
import bpy
import imageio
from tqdm import tqdm
from PIL import Image
import mathutils
import open3d as o3d

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)

from tex_utils_local import (
    dict_to_floor_plan, 
    export_layout_to_mesh_dict_list_v2
)

# Constants
MIN_DIST_OBSTACLE = 0.5  # User requirement
CAMERA_RADIUS = 0.5      # Safety margin for camera body
WORLD_UP = np.array([0, 0, 1])

# Step limits for adaptive sampling
MAX_TRANS_STEP = 0.05
MAX_ROT_STEP = np.radians(0.5)

class CameraPlannerEnv:
    def __init__(
        self,
        room_bounds,
        all_meshes,
        interest_meshes=None,
        collision_samples=100000,
        build_ray_intersector=True,
    ):
        """
        room_bounds: list/array [x_min, y_min, z_min, x_max, y_max, z_max]
        all_meshes: list of trimesh.Trimesh objects (walls, floor, objects)
        interest_meshes: list of trimesh.Trimesh objects (only objects to focus on)
        """
        self.bounds = np.array(room_bounds)
        
        # 1. MERGE MESHES
        if all_meshes:
            self.scene_mesh = trimesh.util.concatenate(all_meshes)
        else:
            self.scene_mesh = trimesh.Trimesh() # Empty mesh if no objects

        print("Finished merging meshes")

        if interest_meshes:
            self.interest_mesh = trimesh.util.concatenate(interest_meshes)
        else:
            self.interest_mesh = trimesh.Trimesh()
        
        print("Finished merging interest meshes")

        # 2. BUILD COLLISION ENGINE (All Meshes)
        self.use_kdtree = True
        
        if len(self.scene_mesh.faces) > 0:
            try:
                # Sample points from the surface (fast) for collision
                self.collision_points, _ = trimesh.sample.sample_surface(
                    self.scene_mesh, collision_samples
                )
                from scipy.spatial import cKDTree
                print(f"Sampled {len(self.collision_points)} collision points, shape: {self.collision_points.shape}")
                self.collision_kdtree = cKDTree(self.collision_points, balanced_tree=False)
                print("Finished building collision kdtree")
            except Exception as e:
                print(f"Warning: Collision point sampling failed ({e}).")
                self.use_kdtree = False
        else:
            self.use_kdtree = False

        print("Finished building collision engine")

        # 3. RAY INTERSECTOR. Panorama placement only performs point-distance
        # checks, so it can skip this relatively expensive structure.
        self.intersector = None
        if build_ray_intersector:
            try:
                from trimesh.ray.ray_pyembree import RayMeshIntersector
                self.intersector = RayMeshIntersector(self.scene_mesh)
            except ImportError:
                from trimesh.ray.ray_triangle import RayMeshIntersector
                self.intersector = RayMeshIntersector(self.scene_mesh)
            print("Finished building ray intersector")

    def is_valid_location(self, point, min_dist=0.1):
        """
        Check if the camera position is inside the room AND 
        far enough from obstacles.
        """
        # A. Room Bounds Check (Simple AABB)
        if not (np.all(point > self.bounds[:3]) and np.all(point < self.bounds[3:])):
            return False

        if not self.use_kdtree:
            return True

        # B. Obstacle Distance Check (Approximate using KDTree of ALL surface points)
        dist, _ = self.collision_kdtree.query(point, k=1)
        
        if dist < min_dist:
             return False
            
        return True

    def is_view_clear(self, origin, target, min_view_dist=0.1):
        """
        Check if the 'forward' ray hits an object too early.
        """
        direction = np.array(target) - np.array(origin)
        dist_to_target = np.linalg.norm(direction)
        
        if dist_to_target < 1e-6: return False # Target too close to origin
        
        direction = direction / dist_to_target
        
        if len(self.scene_mesh.faces) == 0:
            return True

        if self.intersector is None:
            raise RuntimeError("Ray intersector was disabled for this planner")

        # Cast a single ray
        hit_points, _, _ = self.intersector.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction]
        )
        
        if len(hit_points) == 0:
            return True
            
        # Check distance to the closest hit
        closest_hit_dist = np.min(np.linalg.norm(hit_points - origin, axis=1))
        
        # If the ray hits something closer than the target (with some margin), view is blocked
        # Ideally we want to see the target, so if hit < dist_to_target, it's blocked.
        # However, target is on surface, so we might hit target itself.
        if closest_hit_dist < dist_to_target - 0.1:
            return False
            
        return True

    def ray_hits_mesh(self, origin, direction):
        """
        Check if a ray from origin in direction hits the mesh.
        """
        if len(self.scene_mesh.faces) == 0:
            return False

        hit_points, _, _ = self.intersector.intersects_location(
            ray_origins=[origin],
            ray_directions=[direction]
        )
        
        return len(hit_points) > 0

def check_view_up(forward):
    # Check if up vector satisfies constraint: abs(up . world_up) > 0.2
    # forward: (3,)
    right = np.cross(forward, WORLD_UP)
    if np.linalg.norm(right) < 1e-6: 
        return False
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    return abs(np.dot(up, WORLD_UP)) > 0.2

def slerp_vector(v0, v1, t_array):
    # v0, v1: (3,) unit vectors
    # t_array: (N,) or float, 0..1
    # Returns (N, 3)
    
    if np.isscalar(t_array):
        t_array = np.array([t_array])
        
    dot = np.dot(v0, v1)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    
    if abs(sin_theta) < 1e-6:
        # Linear interpolation if parallel
        res = (1 - t_array)[:, np.newaxis] * v0 + t_array[:, np.newaxis] * v1
        norms = np.linalg.norm(res, axis=1)
        # Avoid division by zero
        norms[norms < 1e-6] = 1.0
        return res / norms[:, np.newaxis]
        
    w0 = np.sin((1 - t_array) * theta) / sin_theta
    w1 = np.sin(t_array * theta) / sin_theta
    
    res = w0[:, np.newaxis] * v0 + w1[:, np.newaxis] * v1
    return res

def get_interpolated_forwards(eval_dists, cum_dist, forwards):
    """
    Interpolate forward vectors at given distances along the path.
    """
    new_F = []
    current_seg = 0
    
    # Ensure eval_dists are within range
    eval_dists = np.clip(eval_dists, cum_dist[0], cum_dist[-1])
    
    for d in eval_dists:
        # Find segment
        while current_seg < len(cum_dist) - 2 and d > cum_dist[current_seg+1]:
            current_seg += 1
            
        t_segment_start = cum_dist[current_seg]
        t_segment_end = cum_dist[current_seg+1]
        
        segment_len = t_segment_end - t_segment_start
        if segment_len < 1e-6:
            t = 0.0
        else:
            t = (d - t_segment_start) / segment_len
            t = np.clip(t, 0.0, 1.0)
            
        f0 = forwards[current_seg]
        f1 = forwards[current_seg+1]
        
        res = slerp_vector(f0, f1, t)
        new_F.append(res[0])
        
    return np.array(new_F)

def get_object_view_candidates(env, obj_mesh, num_samples=300):
    """
    Sample candidate views (position, target) around a specific object mesh.
    """
    candidates = []
    
    if obj_mesh is None or obj_mesh.is_empty:
        return candidates

    center = obj_mesh.centroid
    # max_extent = np.max(obj_mesh.bounding_box.extents)
    extent_x = obj_mesh.bounding_box.extents[0] * 0.5
    extent_y = obj_mesh.bounding_box.extents[1] * 0.5
    extent_z = obj_mesh.bounding_box.extents[2] * 0.5
    max_extent = max(extent_x, extent_y, extent_z)
    
    # Sample points on spheres at different radii
    # radii = [1.5 * max_extent, 2.0 * max_extent, 2.5 * max_extent, 3.0 * max_extent]
    
    for _ in range(num_samples):
        # Random direction on sphere
        azimuth = np.random.uniform(0, 2 * np.pi)
        elevation = np.random.uniform(0, 60.0 * np.pi / 180.0)

        direction = np.array([
            np.cos(azimuth) * np.sin(elevation), 
            np.sin(azimuth) * np.sin(elevation), 
            np.cos(elevation)
        ])

        r_direction = direction * max_extent
        scale_r = np.max(np.array([r_direction[0] / extent_x, r_direction[1] / extent_y, r_direction[2] / extent_z]))
        r_direction = r_direction / scale_r

        for _ in range(10):
            r_x = np.random.uniform(1.0, 3.0)
            r_y = np.random.uniform(1.0, 3.0)
            r_z = np.random.uniform(1.0, 3.0)
            r = np.array([r_x, r_y, r_z])
            cam_pos = center + direction * r
            
            # Check if position is valid
            if not env.is_valid_location(cam_pos, min_dist=CAMERA_RADIUS):
                # print(f"Invalid location, too close to obstacles")
                continue
                
            # Vector from camera to object center
            forward = center - cam_pos
            f_norm = np.linalg.norm(forward)
            if f_norm < 1e-6: continue
            forward /= f_norm
            
            # Check constraints
            if not check_view_up(forward):
                # print(f"Invalid view up")
                continue
            
            # if not env.is_view_clear(cam_pos, center):
            #     print(f"Invalid view clear")
            #     continue
            
            candidates.append({
                'position': cam_pos,
                'target': center,
                'forward': forward
            })
            
    return candidates

def generate_anchors(env, room, mesh_dict, max_anchors=20):
    """
    Generate anchor points based on object importance logic.
    """
    
    # 1. Classify objects
    wall_ids = set(w.id for w in room.walls)
    
    # Count occurrences of objects being placed on other objects
    place_counts = Counter()
    children_info = {}
    for obj in room.objects:
        place_counts[obj.place_id] += 1
        children_info[obj.place_id] = children_info.get(obj.place_id, []) + [obj.id]
        
    all_candidates = []
    
    # Weight Constants
    BASE_SCORE_FLOOR = 2.0
    BASE_SCORE_WALL = 1.0
    BONUS_PER_CHILD = 1.0
    
    processed_count = 0
    
    for obj in room.objects:
        # Check if mesh exists
        if obj.id not in mesh_dict:
            continue
            
        weight = 0.0
        is_target = False
        
        if obj.place_id == 'floor':
            weight = BASE_SCORE_FLOOR
            is_target = True
        elif obj.place_id == 'wall':
            weight = BASE_SCORE_WALL
            is_target = True
            
        if is_target:
            # Add bonus for children objects (objects on top)
            weight += place_counts[obj.id] * BONUS_PER_CHILD

            
            # Generate candidates
            mesh_info = mesh_dict.get(obj.id)

            target_mesh = mesh_info['mesh']
            children_meshes = []
            for child_id in children_info.get(obj.id, []):
                child_mesh = mesh_dict.get(child_id)['mesh']
                children_meshes.append(child_mesh)
            
            target_mesh = trimesh.util.concatenate([target_mesh] + children_meshes)

            cands = get_object_view_candidates(env, target_mesh)
            if len(cands) > 2:
                cands = cands[:2]
            for c in cands:
                c['weight'] = weight
                c['obj_id'] = obj.id
                all_candidates.append(c)
            processed_count += 1

            # print(f"Object {obj.id} with weight {weight} and {len(cands)} candidates")
            # if 'vanity' in obj.id:
            #     assert False

            
    print(f"Processed {processed_count} objects for anchors with weighted scoring.")
    
    if not all_candidates:
        print("Warning: No valid object-centric views found. Falling back to random sampling.")
        # Fallback to random valid points
        for _ in range(max_anchors * 5):
             pos = np.random.uniform(env.bounds[:3], env.bounds[3:])
             if env.is_valid_location(pos, min_dist=CAMERA_RADIUS):
                 # Look at random point
                 target = np.random.uniform(env.bounds[:3], env.bounds[3:])
                 fwd = target - pos
                 if np.linalg.norm(fwd) > 0.1:
                     fwd /= np.linalg.norm(fwd)
                     if check_view_up(fwd):
                         all_candidates.append({
                             'position': pos,
                             'target': target,
                             'forward': fwd,
                             'weight': 1.0
                         })
    
    if not all_candidates:
        raise RuntimeError("Could not generate any valid anchors")

    # 3. Select Anchors
    # Probabilistic selection without replacement (if possible)
    selected_anchors = []
    
    weights = np.array([c['weight'] for c in all_candidates])
    weights /= np.sum(weights)
    
    # Use indices
    indices = np.arange(len(all_candidates))
    
    num_select = min(max_anchors, len(all_candidates))
    selected_indices = np.random.choice(indices, size=num_select, replace=False, p=weights)
    
    for idx in selected_indices:
        selected_anchors.append(all_candidates[idx])
        
    return selected_anchors

def connect_anchors_tsp(anchors, start_idx=0):
    """
    Connect anchors using a greedy nearest neighbor approach, considering both translation and rotation.
    """
    if not anchors:
        return []
        
    path = [anchors[start_idx]]
    remaining = anchors[:start_idx] + anchors[start_idx+1:]
    
    current = anchors[start_idx]
    
    # Weights for distance metric
    W_TRANS = 1.0
    W_ROT = 1.5 # 1.5 meter equivalent per radian of rotation
    
    while remaining:
        # Find closest
        best_cost = float('inf')
        best_idx = -1
        
        curr_pos = current['position']
        curr_fwd = current['forward']
        
        for i, cand in enumerate(remaining):
            # Translation distance
            dist_trans = np.linalg.norm(cand['position'] - curr_pos)
            
            # Rotation distance (angle between forward vectors)
            dot = np.dot(curr_fwd, cand['forward'])
            dot = np.clip(dot, -1.0, 1.0)
            dist_rot = np.arccos(dot) # Radians [0, pi]
            
            cost = W_TRANS * dist_trans + W_ROT * dist_rot
            
            if cost < best_cost:
                best_cost = cost
                best_idx = i
        
        current = remaining.pop(best_idx)
        path.append(current)
        
    return path

def generate_smooth_path(anchors, num_frames=300, env=None):
    """
    Interpolate smoothly between anchors using Cubic Hermite Spline.
    """
    if len(anchors) < 2:
        return np.array([anchors[0]['position']]*num_frames), np.array([anchors[0]['forward']]*num_frames)

    positions = np.array([a['position'] for a in anchors])
    forwards = np.array([a['forward'] for a in anchors])
    
    # Add tangents for Catmull-Rom style or just heuristic
    # tangent[i] ~ (P[i+1] - P[i-1]) / 2
    # For start/end, use difference
    n = len(positions)
    tangents = np.zeros_like(positions)
    
    for i in range(n):
        prev_p = positions[max(0, i-1)]
        next_p = positions[min(n-1, i+1)]
        tangents[i] = (next_p - prev_p) * 0.5 # tension
        
    # Create spline
    # Parametrize by cumulative distance
    dists = np.linalg.norm(positions[1:] - positions[:-1], axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    total_dist = cum_dist[-1]
    
    if total_dist < 1e-6:
        return np.resize(positions, (num_frames, 3)), np.resize(forwards, (num_frames, 3))
        
    spline = CubicHermiteSpline(cum_dist, positions, tangents)
    
    # --- Adaptive Sampling Strategy ---
    # 1. Sample densely to estimate complexity
    # Use a high enough resolution to capture curvature
    num_dense = max(num_frames * 10, 2000)
    dense_dists = np.linspace(0, total_dist, num_dense)
    dense_P = spline(dense_dists)
    
    # Interpolate forwards at dense points
    dense_F = get_interpolated_forwards(dense_dists, cum_dist, forwards)
    
    # 2. Compute costs per segment
    # Translation cost
    delta_trans = np.linalg.norm(dense_P[1:] - dense_P[:-1], axis=1)
    
    # Rotation cost
    dot_prods = np.sum(dense_F[1:] * dense_F[:-1], axis=1)
    dot_prods = np.clip(dot_prods, -1.0, 1.0)
    delta_rot = np.arccos(dot_prods)
    
    # Combined cost (normalized by limits)
    # We want step <= LIMIT, so cost = step / LIMIT
    step_costs = np.maximum(delta_trans / MAX_TRANS_STEP, delta_rot / MAX_ROT_STEP)
    
    # Integrate cost to get "effort" coordinate
    cum_effort = np.concatenate(([0], np.cumsum(step_costs)))
    total_effort = cum_effort[-1]
    
    # 3. Generate intermediate high-res path based on effort
    # We want enough frames so that each step is small (<= limits)
    # total_effort is roughly the number of steps needed at limit.
    # Add safety factor and ensure at least num_frames
    ideal_num_frames = int(np.ceil(total_effort * 1.2))
    intermediate_num = max(num_frames, ideal_num_frames)
    
    # Distribute points uniformly in effort space
    target_effort = np.linspace(0, total_effort, intermediate_num)
    
    # Map target effort back to distance
    # interp1d(x=cum_effort, y=dense_dists)
    dist_mapper = interp1d(cum_effort, dense_dists, kind='linear')
    eval_dists = dist_mapper(target_effort)
    
    # Initial intermediate positions
    inter_P = spline(eval_dists)

    # --- Collision Avoidance and Smoothing (on intermediate path) ---
    if env is not None and env.use_kdtree:
        # Check if we have collision info
        # Increase iterations for better convergence with smoothing
        for iteration in range(15): 
            # 1. Check collisions
            dists_to_obs, indices = env.collision_kdtree.query(inter_P)
            
            # Identify violating points
            # Use a slightly larger margin for the path than for static anchors to be safe
            safe_margin = CAMERA_RADIUS + 0.1 
            violations = dists_to_obs < safe_margin
            
            # If no violations and we have done at least one smoothing pass (except if perfectly clean initially)
            if not np.any(violations) and iteration > 0:
                break
                
            # 2. Push points away
            if np.any(violations):
                near_obs_pts = env.collision_points[indices[violations]]
                cam_pts = inter_P[violations]
                
                push_dirs = cam_pts - near_obs_pts
                dirs_norm = np.linalg.norm(push_dirs, axis=1)
                
                # Handle concentric case (rare)
                safe_mask = dirs_norm > 1e-6
                push_dirs[~safe_mask] = np.random.normal(size=(np.sum(~safe_mask), 3))
                push_dirs[~safe_mask] /= np.linalg.norm(push_dirs[~safe_mask], axis=1)[:, np.newaxis]
                dirs_norm[~safe_mask] = 1.0
                
                push_dirs = push_dirs / dirs_norm[:, np.newaxis]
                
                # Push amount: how much deeper are we than safe_margin?
                needed_push = safe_margin - dists_to_obs[violations]
                
                # Add a small buffer to push slightly further to account for smoothing pulling it back
                inter_P[violations] += push_dirs * (needed_push[:, np.newaxis] + 0.05)
            
            # 3. Smooth the path to avoid jaggedness
            # Use Gaussian smoothing for higher quality results
            if len(inter_P) > 5:
                # Apply smoothing
                # Use sigma=2.0 for reasonable smoothness. 
                # Since inter_P is dense (small steps), sigma=2.0 is a local smoothing.
                smoothed_P = gaussian_filter1d(inter_P, sigma=2.0, axis=0, mode='nearest')
                
                # Anchor constraints: keep start/end fixed
                smoothed_P[0] = inter_P[0]
                smoothed_P[-1] = inter_P[-1]
                
                inter_P = smoothed_P
            
            # 4. Enforce room bounds
            inter_P = np.maximum(inter_P, env.bounds[:3] + CAMERA_RADIUS)
            inter_P = np.minimum(inter_P, env.bounds[3:] - CAMERA_RADIUS)
    
    # Calculate intermediate orientations
    inter_F = get_interpolated_forwards(eval_dists, cum_dist, forwards)

    # --- 4. Resample to final num_frames ---
    if intermediate_num == num_frames:
        return inter_P, inter_F
        
    # Resample
    t_inter = np.linspace(0, 1, intermediate_num)
    t_final = np.linspace(0, 1, num_frames)
    
    # Linear interpolation for positions
    resampler_P = interp1d(t_inter, inter_P, axis=0, kind='linear')
    final_P = resampler_P(t_final)
    
    # Linear interpolation for forwards (safe because dense)
    resampler_F = interp1d(t_inter, inter_F, axis=0, kind='linear')
    final_F = resampler_F(t_final)
    
    # Normalize forwards
    norms = np.linalg.norm(final_F, axis=1)
    norms[norms < 1e-6] = 1.0
    final_F = final_F / norms[:, np.newaxis]
    
    return final_P, final_F

def setup_camera_look_at(camera, camera_pos, lookat_pos):
    """Position camera and make it look at target position"""
    # Set camera position
    camera.location = camera_pos
    
    # Calculate direction vector
    direction = mathutils.Vector(lookat_pos) - mathutils.Vector(camera_pos)
    
    # Point camera to look at target
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def generate_camera_trajectory(
    room_bounds, 
    all_meshes, 
    num_frames=300, 
    complexity=10,
    env=None,
    room=None,
    mesh_dict=None
):
    # 0. Preprocessing
    if env is None:
        print("preprocessing: Building environment...")
        env = CameraPlannerEnv(room_bounds, all_meshes)
    
    # 1. Generate Anchors
    anchors = generate_anchors(env, room, mesh_dict, max_anchors=complexity)
    
    # 2. Connect Anchors
    # Start with a random one as the first point
    start_idx = np.random.randint(0, len(anchors))
    sorted_anchors = connect_anchors_tsp(anchors, start_idx)
    
    # 3. Generate Smooth Path
    trajectory_P, trajectory_F = generate_smooth_path(sorted_anchors, num_frames, env=env)
    
    trajectory_poses = []
    
    for i in range(num_frames):
        curr_P = trajectory_P[i]
        curr_F = trajectory_F[i]
        
        # 4. Compute Orientation (LookAt)
        forward = curr_F
        dist = np.linalg.norm(forward)
        if dist < 1e-6: forward = np.array([1, 0, 0])
        else: forward = forward / dist
        
        right = np.cross(forward, WORLD_UP)
        if np.linalg.norm(right) < 1e-6:
             right = np.array([1, 0, 0])
             
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward) 
        up = up / np.linalg.norm(up)
        
        R_mat = np.column_stack([right, up, -forward])
        
        # Calculate lookat target from forward vector
        target_pt = curr_P + forward * 2.0
        
        trajectory_poses.append({
            'position': curr_P,
            'rotation': R_mat,
            'target': target_pt
        })

    return trajectory_poses

def get_room_meshes(layout, layout_dir):
    mesh_info_dict = export_layout_to_mesh_dict_list_v2(layout, layout_dir)
    all_meshes = []
    interest_meshes = []

    # Identify object IDs
    object_ids = set()
    for r in layout.rooms:
        for obj in r.objects:
            object_ids.add(obj.id)

    for mesh_id, mesh_info in mesh_info_dict.items():
        if "mesh" in mesh_info:
            m = mesh_info["mesh"]
            all_meshes.append(m)
            # Check if this mesh corresponds to an object
            if mesh_id in object_ids:
                interest_meshes.append(m)

    print(f"Found {len(all_meshes)} meshes, {len(interest_meshes)} object meshes")
                
    return all_meshes, interest_meshes, mesh_info_dict

def setup_scene_lighting(scene, room_size_dict):
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    world_nodes = scene.world.node_tree.nodes
    world_nodes.clear()
    world_bg = world_nodes.new(type='ShaderNodeBackground')
    world_bg.inputs[0].default_value = (1, 1, 1, 1)
    world_bg.inputs[1].default_value = 1.0
    world_output = world_nodes.new(type='ShaderNodeOutputWorld')
    scene.world.node_tree.links.new(world_output.inputs['Surface'], world_bg.outputs['Background'])
    
    if "WarmAmbientLight" in bpy.data.objects:
        light = bpy.data.objects["WarmAmbientLight"]
    else:
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
        light = bpy.context.active_object
        light.name = "WarmAmbientLight"
    
    light.data.energy = 10.0
    light.data.color = (1.0, 0.9, 0.7)
    light.data.angle = np.radians(10)

    # add ceiling grid lights with 1.0m spacing
    grid_size = 2.0
    offset = 0.1
    ceiling_z = room_size_dict['height'] - offset
    
    # Grid spans from offset to width/length - offset
    x_start = offset
    x_end = room_size_dict['width'] - offset
    y_start = offset
    y_end = room_size_dict['length'] - offset
    
    # Calculate number of lights in each direction
    num_x = max(1, int((x_end - x_start) / grid_size) + 1)
    num_y = max(1, int((y_end - y_start) / grid_size) + 1)
    
    # Calculate spacing
    x_spacing = (x_end - x_start) / max(1, num_x - 1) if num_x > 1 else 0
    y_spacing = (y_end - y_start) / max(1, num_y - 1) if num_y > 1 else 0
    
    for i in range(num_x):
        for j in range(num_y):
            x = x_start + x_spacing * i if num_x > 1 else (x_start + x_end) / 2
            y = y_start + y_spacing * j if num_y > 1 else (y_start + y_end) / 2
            bpy.ops.object.light_add(type='POINT', location=(x, y, ceiling_z))
            light = bpy.context.active_object
            light.name = f"CeilingGridLight_{i}_{j}"
            light.data.energy = 100.0
            light.data.color = (1.0, 0.9, 0.7)

def get_or_create_collection(collection_name):
    """Get or create a collection"""
    if collection_name in bpy.data.collections:
        return bpy.data.collections[collection_name]
    
    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def get_material_pbr_factors(texture_info):
    """Return normalized metallic/roughness factors from SAGE texture metadata."""
    pbr_parameters = texture_info.get("pbr_parameters") or {}
    metallic = texture_info.get(
        "metallic_factor", pbr_parameters.get("metallic", 0.0)
    )
    roughness = texture_info.get(
        "roughness_factor", pbr_parameters.get("roughness", 1.0)
    )
    return float(metallic), float(roughness)


def clear_blender_scene():
    """Clear all objects from Blender scene"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    
    # Clear all collections except the default Scene Collection
    for collection in list(bpy.data.collections):
        bpy.data.collections.remove(collection)


def load_scene_meshes_into_blender(room, layout, layout_dir, mesh_info_dict=None):
    """Load room layout meshes from files into Blender"""
    
    # Clear all existing Blender assets before loading new ones
    clear_blender_scene()
    
    # Get mesh info dict
    if mesh_info_dict is None:
        mesh_info_dict = export_layout_to_mesh_dict_list_v2(layout, layout_dir)
    
    # Create collection for scene objects
    scene_collection = get_or_create_collection("scene_objects")
    
    # Identify object IDs for pass index assignment
    # Sort for determinism
    object_ids = sorted([obj.id for obj in room.objects])
    obj_id_to_pass_index = {oid: idx + 1 for idx, oid in enumerate(object_ids)}
    print(f"Assigned pass indices for {len(object_ids)} objects (indices 1-{len(object_ids)})")

    # Import each mesh
    for mesh_id, mesh_info in mesh_info_dict.items():
        # if mesh_id.startswith("door") or mesh_id.startswith("window"):
        #     continue
        
        trimesh_mesh = mesh_info["mesh"]
        
        # Convert trimesh to Blender mesh
        vertices = trimesh_mesh.vertices
        faces = trimesh_mesh.faces
        
        # Create new mesh data
        mesh_data = bpy.data.meshes.new(name=f"mesh_{mesh_id}")
        mesh_data.from_pydata(vertices.tolist(), [], faces.tolist())
        mesh_data.update()
        
        # Create object from mesh
        obj = bpy.data.objects.new(mesh_id, mesh_data)
        
        # Assign pass index
        if mesh_id in obj_id_to_pass_index:
            obj.pass_index = obj_id_to_pass_index[mesh_id]
        else:
            obj.pass_index = 0

        scene_collection.objects.link(obj)
        
        # Load and apply texture if available
        texture_info = mesh_info.get("texture")
        if texture_info and texture_info.get("texture_map_path"):
            texture_path = texture_info["texture_map_path"]
            if os.path.exists(texture_path):
                # Create material with texture
                mat = bpy.data.materials.new(name=f"mat_{mesh_id}")
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                nodes.clear()
                
                # Create shader nodes
                bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
                tex_image = nodes.new(type='ShaderNodeTexImage')
                output = nodes.new(type='ShaderNodeOutputMaterial')
                
                # Load texture image
                tex_image.image = bpy.data.images.load(texture_path)
                tex_image.image.colorspace_settings.name = 'sRGB'  # Ensure correct color space

                # Object PBR values are nested under ``pbr_parameters`` in
                # SAGE layouts, while some older assets use top-level factors.
                # Metallic and specular are distinct Principled BSDF inputs;
                # assigning metallic to specular makes metals render as matte
                # dielectrics.
                metallic, roughness = get_material_pbr_factors(texture_info)
                bsdf.inputs['Roughness'].default_value = roughness
                metallic_input = bsdf.inputs.get('Metallic')
                if metallic_input is not None:
                    metallic_input.default_value = metallic
                sheen_tint = bsdf.inputs.get('Sheen Tint')
                if sheen_tint is not None:
                    # Blender 3/4 exposed a scalar here, while Blender 5 uses
                    # an RGBA color socket.
                    if getattr(sheen_tint, 'type', None) == 'RGBA':
                        sheen_tint.default_value = (0.0, 0.0, 0.0, 1.0)
                    else:
                        sheen_tint.default_value = 0.0
                
                # Connect nodes
                mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
                mat.node_tree.links.new(output.inputs['Surface'], bsdf.outputs['BSDF'])
                
                # Apply material to object
                if obj.data.materials:
                    obj.data.materials[0] = mat
                else:
                    obj.data.materials.append(mat)
                
                # Set UV coordinates if available
                vts = texture_info.get("vts")
                fts = texture_info.get("fts")
                if vts is not None and fts is not None:
                    # Create UV layer
                    uv_layer = obj.data.uv_layers.new(name="UVMap")
                    for face_idx, face in enumerate(fts):
                        for vert_idx in range(len(face)):
                            loop_idx = face_idx * len(face) + vert_idx
                            if loop_idx < len(uv_layer.data):
                                uv = vts[face[vert_idx]]
                                uv_layer.data[loop_idx].uv = (uv[0], uv[1])
    
    print(f"Loaded {len(mesh_info_dict)} meshes into Blender scene")
    return mesh_info_dict, len(object_ids)



def render_trajectory_video(layout, layout_dir, room_id, trajectory, output_path, fov=30.0, resolution=512, fps=30, render_depth=True):

    
    all_rooms = layout.rooms
    room = next((r for r in all_rooms if r.id == room_id), None)
    if room is None:
        raise ValueError(f"Room {room_id} not found in layout")
    
    print("Loading scene meshes into Blender...")
    mesh_info_dict, num_objects = load_scene_meshes_into_blender(room, layout, layout_dir)
    
    scene_collection = bpy.data.collections.get("scene_objects")
    scene_objects = list(scene_collection.objects) if scene_collection else []
    
    scene = bpy.context.scene
    
    if "Camera" in bpy.data.objects:
        camera = bpy.data.objects["Camera"]
    else:
        bpy.ops.object.camera_add()
        camera = bpy.context.active_object
        camera.name = "Camera"
    scene.camera = camera
    
    camera.data.type = 'PERSP'
    camera.data.sensor_fit = 'VERTICAL'
    camera.data.angle = np.radians(fov)

    if args.engine == 'CYCLES':
    
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 64  # Lower samples, denoiser compensates
        scene.cycles.use_denoising = True
        
        # Enable GPU/CUDA rendering
        scene.cycles.device = 'GPU'

        # Use OptiX denoiser (GPU-accelerated, much faster than OIDN on NVIDIA)
        # Fall back to OIDN if OptiX not available
        # scene.cycles.denoiser = 'OPENIMAGEDENOISE'

        preferences = bpy.context.preferences
        cycles_preferences = preferences.addons['cycles'].preferences
        cycles_preferences.compute_device_type = 'OPTIX'
        # scene.cycles.denoiser = 'OPTIX'
        # Refresh devices to ensure L40S is recognized
        cycles_preferences.get_devices()
        for device in cycles_preferences.devices:
            device.use = (device.type == 'OPTIX')
        
        # GPU performance optimizations
        scene.cycles.use_adaptive_sampling = True  # Reduces samples in simple areas
        scene.cycles.adaptive_threshold = 0.01
        scene.render.use_persistent_data = True  # Keep BVH in memory between frames
        
        # Reduce light bounces for faster rendering (still good quality for interiors)
        scene.cycles.max_bounces = 2
        scene.cycles.diffuse_bounces = 2
        scene.cycles.glossy_bounces = 1
        scene.cycles.transmission_bounces = 1
        scene.cycles.volume_bounces = 0
        scene.cycles.transparent_max_bounces = 1
    
    else:
        scene.render.engine = 'BLENDER_EEVEE'
        scene.eevee.taa_render_samples = 64
        scene.eevee.use_gtao = True
        scene.eevee.use_ssr = True
        scene.eevee.use_soft_shadows = True

    room_size_dict = {
        'width': room.dimensions.width,
        'length': room.dimensions.length,
        'height': room.dimensions.height
    }
    
    setup_scene_lighting(scene, room_size_dict)
    
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    
    scene.use_nodes = True
    scene.view_layers["ViewLayer"].use_pass_combined = True
    scene.view_layers["ViewLayer"].use_pass_object_index = True
    scene.view_layers["ViewLayer"].use_pass_z = True  # Enable depth pass

    output_dir = os.path.dirname(output_path)
    output_basename = os.path.splitext(os.path.basename(output_path))[0]
    frames_dir = os.path.join(output_dir, f"{output_basename}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Setup Compositor for Instance Masks
    tree = scene.node_tree
    # Clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)
        
    rl_node = tree.nodes.new('CompositorNodeRLayers')
    
    # Output node for masks
    masks_dir = os.path.join(output_dir, f"{output_basename}_masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    mask_output_node = tree.nodes.new('CompositorNodeOutputFile')
    mask_output_node.base_path = masks_dir
    mask_output_node.file_slots[0].path = "mask_"
    mask_output_node.file_slots[0].use_node_format = True
    mask_output_node.file_slots[0].format.file_format = 'PNG'
    mask_output_node.file_slots[0].format.color_depth = '8' 
    mask_output_node.file_slots[0].format.color_mode = 'BW'
    
    # Divide index by 255 so that ID 1 becomes 1/255 (value 1 in 8-bit PNG)
    math_node = tree.nodes.new('CompositorNodeMath')
    math_node.operation = 'DIVIDE'
    math_node.inputs[1].default_value = 255.0
    
    tree.links.new(rl_node.outputs['IndexOB'], math_node.inputs[0])
    tree.links.new(math_node.outputs[0], mask_output_node.inputs[0])
    
    # Setup depth output via File Output node with EXR format
    depth_dir = None
    if render_depth:
        depth_dir = os.path.join(output_dir, f"{output_basename}_depth")
        os.makedirs(depth_dir, exist_ok=True)
        
        # Add File Output node for depth - saves as EXR (32-bit float)
        depth_output_node = tree.nodes.new('CompositorNodeOutputFile')
        depth_output_node.name = "DepthOutput"
        depth_output_node.base_path = depth_dir
        depth_output_node.file_slots[0].path = "depth_"
        depth_output_node.file_slots[0].use_node_format = False
        depth_output_node.file_slots[0].format.file_format = 'OPEN_EXR'
        depth_output_node.file_slots[0].format.color_depth = '32'
        tree.links.new(rl_node.outputs['Depth'], depth_output_node.inputs[0])
    
    rendered_frames = []
    depth_frames = []  # Store depth arrays for video generation
    
    print(f"Rendering {len(trajectory)} frames...")
    for i, pose in enumerate(tqdm(trajectory, desc="Rendering frames")):
        scene.frame_set(i) # Update frame for File Output node naming
        
        camera_pos = pose['position']
        lookat_pos = pose['target']
        
        setup_camera_look_at(camera, camera_pos, lookat_pos)
        
        frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
        scene.render.filepath = frame_path
        
        # Render the frame
        with suppress_output():
            bpy.ops.render.render(write_still=True)
        
        if os.path.exists(frame_path):
            img = Image.open(frame_path)
            frame_img = np.array(img)
            
            if frame_img.shape[-1] == 4:
                alpha_mask = frame_img[:, :, 3] == 0
                frame_img[alpha_mask, 0:3] = 255
            
            rendered_frames.append(frame_img)
        
        # Depth EXR is saved by the compositor File Output node
        # We'll read all depth files after rendering is complete
            
    print(f"Saving video to {output_path}...")
    imageio.mimsave(output_path, rendered_frames, fps=fps)
    
    # --- Generate Mask Video ---
    mask_files = []
    if os.path.exists(masks_dir):
        # List all png files in masks_dir
        # Expected format: mask_XXXX.png (created by blender file output node)
        # We need to sort them numerically to ensure correct order
        files = [f for f in os.listdir(masks_dir) if f.endswith('.png')]
        
        # Helper to extract frame number
        def get_frame_num(fname):
            # Assumes format mask_XXXX.png or similar where last part is number
            name_part = os.path.splitext(fname)[0]
            # Try to find the number at the end
            try:
                return int(name_part.split('_')[-1])
            except ValueError:
                return 0
                
        files.sort(key=get_frame_num)
        mask_files = [os.path.join(masks_dir, f) for f in files]
        
    if mask_files and len(mask_files) == len(trajectory):
        print("Generating mask video with random colors...")
        mask_frames = []
        
        # Generate stable random colors
        # 0 is background (black)
        # 1..num_objects are instances
        np.random.seed(42)
        # Ensure at least enough colors
        # +1 for background, +1 for safety margin
        palette_size = max(num_objects + 2, 256)
        colors = np.random.randint(50, 256, size=(palette_size, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0] # Background Black
        
        for fpath in tqdm(mask_files, desc="Processing masks"):
            try:
                # Read mask image
                img = Image.open(fpath)
                m = np.array(img)
                
                # Check dimensions. If saved as RGB/RGBA by mistake, take first channel
                if m.ndim == 3:
                    m = m[..., 0]
                
                # m contains values 0..N (where 0 is bg, 1..N are objects)
                # Clip to palette size just in case
                m_indices = np.clip(m, 0, palette_size - 1)
                
                # Map to colors
                colored_frame = colors[m_indices]
                mask_frames.append(colored_frame)
            except Exception as e:
                print(f"Warning: Failed to process mask {fpath}: {e}")
                
        if mask_frames:
            mask_video_path = output_path.replace('.mp4', '_mask.mp4')
            # Handle case where extension wasn't .mp4
            if mask_video_path == output_path:
                 mask_video_path = output_path + '_mask.mp4'
                 
            print(f"Saving mask video to {mask_video_path}...")
            imageio.mimsave(mask_video_path, mask_frames, fps=fps)

    # --- Process Depth EXR files and Generate Visualization Video ---
    if render_depth and depth_dir and os.path.exists(depth_dir):
        print("Processing depth EXR files using Blender...")
        import matplotlib.pyplot as plt
        
        # Find all EXR files in depth_dir
        exr_files = [f for f in os.listdir(depth_dir) if f.endswith('.exr')]
        
        # Sort by frame number
        def get_frame_num(fname):
            name_part = os.path.splitext(fname)[0]
            try:
                return int(name_part.split('_')[-1])
            except ValueError:
                return 0
        
        exr_files.sort(key=get_frame_num)
        print(f"Found {len(exr_files)} EXR depth files")
        
        if exr_files:
            depth_frames = []
            
            # Read all EXR files using Blender's image loading
            for exr_fname in tqdm(exr_files, desc="Reading depth EXR files"):
                exr_path = os.path.join(depth_dir, exr_fname)
                
                # Load image using Blender
                img = bpy.data.images.load(exr_path)
                
                width = img.size[0]
                height = img.size[1]
                
                # Get pixel data as float array
                pixels = np.array(img.pixels[:], dtype=np.float32)
                pixels = pixels.reshape((height, width, 4))  # RGBA
                
                # Depth is in R channel
                depth = pixels[:, :, 0].copy()
                
                # Flip vertically (Blender images are bottom-up)
                depth = np.flipud(depth)
                
                # print(f"Depth: {depth.shape}, {depth.min():.4f}, {depth.max():.4f}")
                
                depth_frames.append(depth)
                
                # Save as .npy
                frame_num = get_frame_num(exr_fname)
                npy_path = os.path.join(depth_dir, f"depth_{frame_num:04d}.npy")
                np.save(npy_path, depth)
                
                # Remove from Blender to free memory
                bpy.data.images.remove(img)
            
            # Remove EXR files after converting to npy
            for exr_fname in exr_files:
                exr_path = os.path.join(depth_dir, exr_fname)
                os.remove(exr_path)
            
            print("Generating depth visualization video with plasma colormap...")
            
            # Find global max depth for normalization (excluding inf values)
            all_depths = np.concatenate([d.flatten() for d in depth_frames])
            finite_depths = all_depths[np.isfinite(all_depths) & (all_depths > 0) & (all_depths < 1e3)]
            if len(finite_depths) > 0:
                global_max_depth = np.max(finite_depths)
            else:
                global_max_depth = 1.0

            print(f"Global max depth: {global_max_depth}")
            
            depth_viz_frames = []
            plasma_cmap = plt.cm.plasma
            
            for depth in tqdm(depth_frames, desc="Processing depth frames"):
                # Handle infinite values (background/sky)
                depth_viz = depth.copy()
                depth_viz[~np.isfinite(depth_viz)] = global_max_depth
                
                # Normalize by max depth
                depth_normalized = depth_viz / global_max_depth
                depth_normalized = np.clip(depth_normalized, 0, 1)
                
                # Apply plasma colormap
                depth_colored = plasma_cmap(depth_normalized)
                
                # Convert to uint8 RGB (drop alpha channel)
                depth_rgb = (depth_colored[:, :, :3] * 255).astype(np.uint8)
                depth_viz_frames.append(depth_rgb)
            
            depth_video_path = output_path.replace('.mp4', '_depth.mp4')
            if depth_video_path == output_path:
                depth_video_path = output_path + '_depth.mp4'
            
            print(f"Saving depth video to {depth_video_path}...")
            imageio.mimsave(depth_video_path, depth_viz_frames, fps=fps)
            
            print(f"Depth .npy files saved to {depth_dir}")

    print("Done!")


def save_point_cloud(output_path, camera_json_path, voxel_size=0.02, save_instance_cloud=True):
    """
    Create a merged point cloud from RGB-D frames using Open3D.
    
    Args:
        output_path: Path to save the .ply file
        camera_json_path: Path to the camera trajectory JSON file containing K, width, height, and frames
        voxel_size: Voxel size for downsampling (default 0.02m = 2cm)
        save_instance_cloud: If True, also save an instance-colored point cloud (bg=white, instances=random colors)
    """
    # Load camera data from JSON
    with open(camera_json_path, 'r') as f:
        camera_data = json.load(f)
    
    K = np.array(camera_data['K'])
    width = camera_data['width']
    height = camera_data['height']
    frames = camera_data['frames']
    
    # Derive directory paths from camera JSON path (assumes video is .mp4 with same basename)
    output_dir = os.path.dirname(camera_json_path)
    output_basename = os.path.splitext(os.path.basename(camera_json_path))[0]
    
    frames_dir = os.path.join(output_dir, f"{output_basename}_frames")
    depth_dir = os.path.join(output_dir, f"{output_basename}_depth")
    masks_dir = os.path.join(output_dir, f"{output_basename}_masks")
    
    if not os.path.exists(frames_dir):
        print(f"Error: Frames directory not found: {frames_dir}")
        return
    if not os.path.exists(depth_dir):
        print(f"Error: Depth directory not found: {depth_dir}")
        return
    
    # Check if masks directory exists for instance cloud
    has_masks = os.path.exists(masks_dir) and save_instance_cloud
    if save_instance_cloud and not os.path.exists(masks_dir):
        print(f"Warning: Masks directory not found: {masks_dir}. Skipping instance point cloud.")
    
    # Create Open3D intrinsic object
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2]
    )
    
    # Find all frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
    mask_files = sorted([f for f in os.listdir(masks_dir) if f.endswith('.png')]) if has_masks else []
    
    print(f"Found {len(frame_files)} RGB frames and {len(depth_files)} depth maps")
    if has_masks:
        print(f"Found {len(mask_files)} mask files")
    
    if len(frame_files) != len(frames) or len(depth_files) != len(frames):
        print(f"Warning: Mismatch in counts - rgb_frames: {len(frame_files)}, "
              f"depths: {len(depth_files)}, camera_frames: {len(frames)}")
    
    # Use minimum count
    num_frames = min(len(frame_files), len(depth_files), len(frames))
    if has_masks:
        num_frames = min(num_frames, len(mask_files))
    
    # Generate random colors for instances (consistent across frames)
    # Use a fixed seed for reproducibility
    np.random.seed(42)
    palette_size = 256  # Should be enough for most scenes
    instance_colors = np.random.randint(50, 256, size=(palette_size, 3), dtype=np.uint8)
    instance_colors[0] = [255, 255, 255]  # Background is white
    
    # Merged point clouds
    merged_pcd = o3d.geometry.PointCloud()
    merged_instance_pcd = o3d.geometry.PointCloud() if has_masks else None
    
    print(f"Processing {num_frames} frames...")
    for i in tqdm(range(num_frames), desc="Building point cloud"):
        # Load RGB image
        rgb_path = os.path.join(frames_dir, frame_files[i])
        rgb_img = np.array(Image.open(rgb_path))
        
        # Handle RGBA -> RGB
        if rgb_img.shape[-1] == 4:
            rgb_img = rgb_img[:, :, :3]
        
        # Load depth map
        depth_path = os.path.join(depth_dir, depth_files[i])
        depth = np.load(depth_path)
        
        # Handle infinite/invalid depth values
        max_depth = 100.0  # Clip to reasonable max
        depth[~np.isfinite(depth)] = 0
        depth[depth > max_depth] = 0
        
        # Get camera pose from JSON frame data (eye, lookat, up)
        frame_data = frames[i]
        eye = np.array(frame_data['eye'])
        lookat = np.array(frame_data['lookat'])
        up_vec = np.array(frame_data['up'])
        
        # Reconstruct rotation matrix from eye, lookat, up
        # forward = direction camera is looking
        forward = lookat - eye
        forward = forward / np.linalg.norm(forward)
        
        # right = forward × up (perpendicular to both)
        right = np.cross(forward, up_vec)
        right = right / np.linalg.norm(right)
        
        # Recompute up to ensure orthonormality
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Build rotation matrix: columns are [right, -up, forward]
        R_mat = np.column_stack([right, -up, forward])
        
        # Build camera-to-world transform (extrinsic)
        # R_mat transforms from camera to world: world_vec = R_mat @ cam_vec
        extrinsic_c2w = np.eye(4)
        extrinsic_c2w[:3, :3] = R_mat
        extrinsic_c2w[:3, 3] = eye
        
        # Open3D's create_point_cloud_from_rgbd_image expects world-to-camera (extrinsic)
        # So we need the inverse
        extrinsic_w2c = np.linalg.inv(extrinsic_c2w)
        
        # Convert to Open3D images for RGB point cloud
        rgb_o3d = o3d.geometry.Image(rgb_img.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=1.0,  # depth is already in meters
            depth_trunc=max_depth,
            convert_rgb_to_intensity=False
        )
        
        # Create point cloud from this frame
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            intrinsic,
            extrinsic=extrinsic_w2c
        )
        
        # Merge into main point cloud
        merged_pcd += pcd
        
        # Process instance mask if available
        if has_masks:
            # Load mask
            mask_path = os.path.join(masks_dir, mask_files[i])
            mask_img = np.array(Image.open(mask_path))
            
            # Handle RGB/RGBA mask - take first channel
            if mask_img.ndim == 3:
                mask_img = mask_img[..., 0]
            
            # Map mask indices to colors
            mask_indices = np.clip(mask_img, 0, palette_size - 1)
            instance_rgb = instance_colors[mask_indices]
            
            # Create instance-colored RGBD image
            instance_rgb_o3d = o3d.geometry.Image(instance_rgb.astype(np.uint8))
            depth_o3d_instance = o3d.geometry.Image(depth.astype(np.float32))
            
            rgbd_instance = o3d.geometry.RGBDImage.create_from_color_and_depth(
                instance_rgb_o3d,
                depth_o3d_instance,
                depth_scale=1.0,
                depth_trunc=max_depth,
                convert_rgb_to_intensity=False
            )
            
            # Create instance point cloud from this frame
            instance_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_instance,
                intrinsic,
                extrinsic=extrinsic_w2c
            )
            
            merged_instance_pcd += instance_pcd
    
    print(f"Total points before downsampling: {len(merged_pcd.points)}")
    
    # Downsample using voxel grid
    if voxel_size > 0 and len(merged_pcd.points) > 0:
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Total points after downsampling (voxel_size={voxel_size}m): {len(merged_pcd.points)}")
    
    # Remove statistical outliers
    if len(merged_pcd.points) > 100:
        merged_pcd, _ = merged_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"Total points after outlier removal: {len(merged_pcd.points)}")
    
    # Save point cloud
    o3d.io.write_point_cloud(output_path, merged_pcd)
    print(f"Point cloud saved to {output_path}")
    
    # Process and save instance point cloud
    if has_masks and merged_instance_pcd is not None:
        print(f"Instance point cloud - Total points before downsampling: {len(merged_instance_pcd.points)}")
        
        if voxel_size > 0 and len(merged_instance_pcd.points) > 0:
            merged_instance_pcd = merged_instance_pcd.voxel_down_sample(voxel_size=voxel_size)
            print(f"Instance point cloud - Total points after downsampling: {len(merged_instance_pcd.points)}")
        
        if len(merged_instance_pcd.points) > 100:
            merged_instance_pcd, _ = merged_instance_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            print(f"Instance point cloud - Total points after outlier removal: {len(merged_instance_pcd.points)}")
        
        # Generate instance output path
        instance_output_path = output_path.replace('.ply', '_instance.ply')
        if instance_output_path == output_path:
            instance_output_path = output_path + '_instance.ply'
        
        o3d.io.write_point_cloud(instance_output_path, merged_instance_pcd)
        print(f"Instance point cloud saved to {instance_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate camera trajectory for a room")
    parser.add_argument("layout_path", type=str, help="Layout ID to visualize")
    parser.add_argument("--room_id", type=str, help="Room ID (optional, defaults to first room)")
    parser.add_argument("--frames", type=int, default=300, help="Number of frames")
    parser.add_argument("--complexity", type=int, default=25, help="Number of anchors")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="trajectory.json", help="Output JSON path")
    parser.add_argument("--render", action="store_true", help="Render video using Blender")
    parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"], help="Render engine")
    parser.add_argument("--resolution", type=int, default=1024, help="Resolution width (if rendering)")
    parser.add_argument("--fps", type=int, default=30, help="FPS (if rendering)")
    parser.add_argument("--save_point_cloud", action="store_true", help="Save merged point cloud from RGB-D frames")
    
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = sys.argv[1:]
    
    args = parser.parse_args(argv)
    
    try:
        layout_dir = os.path.dirname(args.layout_path)
        json_path = args.layout_path
        
        if not os.path.exists(json_path):
             print(f"Error: Layout file not found at {json_path}")
             sys.exit(1)
             
        print(f"Loading layout from {json_path}...")
        with open(json_path, 'r') as f:
            layout_data = json.load(f)
        layout = dict_to_floor_plan(layout_data)
        
        if args.room_id:
            room = next((r for r in layout.rooms if r.id == args.room_id), None)
            if room is None:
                print(f"Error: Room {args.room_id} not found")
                sys.exit(1)
        else:
            room = layout.rooms[0]
            print(f"Selected room: {room.id}")
            
        bounds = [
            room.position.x, room.position.y, room.position.z,
            room.position.x + room.dimensions.width,
            room.position.y + room.dimensions.length,
            room.position.z + room.dimensions.height
        ]
        
        print("Extracting meshes...")
        all_meshes, interest_meshes, mesh_info_dict = get_room_meshes(layout, layout_dir)
        
        print("Building environment...")
        env = CameraPlannerEnv(bounds, all_meshes, interest_meshes)

        base_output, ext = os.path.splitext(args.output)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        if ext == '': ext = '.json'

        for sample_idx in range(args.num_samples):
            print(f"\n--- Generating sample {sample_idx} / {args.num_samples} ---")
            trajectory = generate_camera_trajectory(
                bounds,
                all_meshes,
                num_frames=args.frames,
                complexity=args.complexity,
                env=env,
                room=room,
                mesh_dict=mesh_info_dict
            )
            
            # Compute camera intrinsic matrix K
            args.fov = np.random.choice([35.0, 45.0, 60.0, 80.0])
            fov_y = np.radians(args.fov)  # Vertical FOV from camera settings
            width = args.resolution
            height = width  # Aspect ratio from render settings
            
            fy = height / (2 * np.tan(fov_y / 2))
            fx = fy  # Square pixels
            cx = width / 2.0
            cy = height / 2.0
            
            K = [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ]
            
            frames_data = []
            for pose in trajectory:
                # up vector is column 1 of rotation matrix (R_mat = [right, up, -forward])
                up_vector = pose['rotation'][:, 1]
                
                frames_data.append({
                    'eye': pose['position'].tolist(),
                    'lookat': pose['target'].tolist(),
                    'up': up_vector.tolist()
                })
            
            output_data = {
                'K': K,
                'width': width,
                'height': height,
                'fov_y_deg': args.fov,
                'frames': frames_data
            }
            
            current_output_path = f"{base_output}_{sample_idx}{ext}"
            
            with open(current_output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
            print(f"Trajectory saved to {current_output_path} with {len(frames_data)} frames")
            
            if args.render:
                print(f"Rendering video for sample {sample_idx}...")
                video_output_path = current_output_path.replace('.json', '.mp4')
                if video_output_path == current_output_path:
                    video_output_path = current_output_path + '.mp4'
                    
                render_trajectory_video(
                    layout, 
                    layout_dir,
                    room.id, 
                    trajectory, 
                    video_output_path, 
                    fov=args.fov,
                    resolution=args.resolution,
                    fps=args.fps,
                    render_depth=True
                )

            if args.save_point_cloud:
                if not args.render:
                    print("Warning: --save_point_cloud requires --render to generate RGB-D frames. Skipping.")
                else:
                    print(f"Saving point cloud for sample {sample_idx}...")
                    point_cloud_output_path = current_output_path.replace('.json', '_point_cloud.ply')
                    if point_cloud_output_path == current_output_path:
                        point_cloud_output_path = current_output_path + '_point_cloud.ply'
                    
                    save_point_cloud(
                        output_path=point_cloud_output_path,
                        camera_json_path=current_output_path
                    )
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
