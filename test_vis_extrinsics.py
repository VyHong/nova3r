import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_extrinsics(extrinsics, save_path="extrinsics_vis.png"):
    # extrinsics: [S, 4, 4]
    if isinstance(extrinsics, torch.Tensor):
        extrinsics = extrinsics.detach().cpu().numpy()
        
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.jet(np.linspace(0, 1, len(extrinsics)))
    
    for i, ext in enumerate(extrinsics):
        # Assuming extrinsics is world-to-camera matrix
        R = ext[:3, :3]
        t = ext[:3, 3]
        
        # Camera center in world coordinates
        cam_center = -R.T @ t
        # Or if it's camera-to-world
        # cam_center = t
        
        # We plot both to be safe or determine it based on standard conventions
        # Usually it's w2c, so cam_center is -R.T @ t. If it's c2w, R is already camera orientation, t is already center.
        # Let's plot t directly if it's c2w. Wait, typical "extrinsics" is w2c. 
        # I'll just save it to a file.
        
        ax.scatter(cam_center[0], cam_center[1], cam_center[2], color=colors[i], label=f'Cam {i}')
        
        # plot camera axis
        # Z axis is usually R.T @ [0,0,1]
        z_dir = R.T @ np.array([0, 0, 1])
        ax.quiver(cam_center[0], cam_center[1], cam_center[2], z_dir[0], z_dir[1], z_dir[2], length=0.1, color='b')
        x_dir = R.T @ np.array([1, 0, 0])
        ax.quiver(cam_center[0], cam_center[1], cam_center[2], x_dir[0], x_dir[1], x_dir[2], length=0.1, color='r')
        y_dir = R.T @ np.array([0, 1, 0])
        ax.quiver(cam_center[0], cam_center[1], cam_center[2], y_dir[0], y_dir[1], y_dir[2], length=0.1, color='g')
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved visualization to {save_path}")

