import torch
import time

# Configuration
TARGET_UTIL = 0.80  # 20%
MATRIX_SIZE = 4096  # Adjust this if load is too low/high
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Targeting ~{TARGET_UTIL*100}% utilization on {device}...")
print("Monitor with: watch -n 1 nvidia-smi")

# Initialize small matrices (Uses ~67MB of VRAM)
a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
b = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)

try:
    while True:
        start_time = time.time()
        
        # Perform a burst of work
        # Running it multiple times ensures the GPU doesn't finish too fast
        for _ in range(10):
            c = torch.matmul(a, b)
        
        # Synchronize to ensure the GPU is actually finished
        torch.cuda.synchronize()
        
        work_duration = time.time() - start_time
        
        # Calculate required sleep to maintain 20% duty cycle
        # Work / (Work + Sleep) = 0.20  => Sleep = (Work / 0.20) - Work
        sleep_duration = (work_duration / TARGET_UTIL) - work_duration
        
        if sleep_duration > 0:
            time.sleep(sleep_duration)
            
except KeyboardInterrupt:
    print("\nLoad stopped.")