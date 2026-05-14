
import numpy as np
import torch
import os

noise_path = "fixed_noise_16k.npy"

if not os.path.exists(noise_path):
    # Generate once and save
    # Shape: (1, 8000, 3)
    initial_noise = np.random.randn(1, 16384, 3).astype(np.float32)
    np.save(noise_path, initial_noise)
    print(f"Saved new noise to {noise_path}")

# Load and convert to Torch
noise_np = np.load(noise_path)
fixed_noise = torch.from_numpy(noise_np).cuda()