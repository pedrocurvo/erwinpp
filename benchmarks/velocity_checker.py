import torch
import time
import sys

# Add the parent directory to the Python path using ..
sys.path.append("..")

from models.erwin import ErwinTransformer
from models.erwin_flash import ErwinTransformer as ErwinTransformerFlash
import csv
import os

output_file = "erwin_benchmark.csv"
write_header = not os.path.exists(output_file)

csv_file = open(output_file, mode="a", newline="")
csv_writer = csv.writer(csv_file)

if write_header:
    csv_writer.writerow([
        "model", "num_points", "batch_size",
        "time_sec", "mem_mb", "output_shape"
    ])

# Function to benchmark any model
@torch.no_grad()
def benchmark_model(model, name, num_points, batch_size, use_amp=False, amp_dtype=torch.bfloat16):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    start = time.time()
    if use_amp:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            output = model(node_features, node_positions, batch_idx, radius=radius)
    else:
        output = model(node_features, node_positions, batch_idx, radius=radius)
    torch.cuda.synchronize()
    end = time.time()

    mem_used = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    time_elapsed = end - start
    shape_str = str(list(output.shape))

    print(f"{name:>24} | Points: {num_points:6} | Time: {time_elapsed:.4f}s | Mem: {mem_used:.1f} MB | Output: {shape_str}")

    csv_writer.writerow([name, num_points, batch_size, f"{time_elapsed:.4f}", f"{mem_used:.1f}", shape_str])
    csv_file.flush()

# Configuration for both models
config = {
    'c_in': 32,
    'c_hidden': [32, 64, 128, 256, 512],
    'enc_num_heads': [2, 4, 8, 16, 32],
    'enc_depths': [2, 2, 2, 6, 2],
    'dec_num_heads': [4, 4, 8, 16],
    'dec_depths': [2, 2, 2, 2],
    'strides': [2, 2, 2, 2],
    'ball_sizes': [128, 128, 128, 128, 128],
    'dimensionality': 3,  # 3D space
    'mlp_ratio': 4,
    'rotate': 0,  # disable cross-ball interaction
}

# Initialize both models
model = ErwinTransformer(**config).cuda().eval()
flash_model = ErwinTransformerFlash(**config).cuda().eval()

for num_points in [1000 * i for i in range(1, 101)]:
    # Create dummy input batch
    batch_size = 16
    c_in = config['c_in']

    node_features = torch.randn(num_points * batch_size, c_in, device="cuda")
    node_positions = torch.rand(num_points * batch_size, 3, device="cuda")
    batch_idx = torch.repeat_interleave(torch.arange(batch_size, device="cuda"), num_points)
    radius = 0.05  # radius for neighborhood connectivity

    # Print the number of points
    print(f"\nNumber of points: {num_points}")
    benchmark_model(model, "ErwinTransformer", num_points, batch_size, use_amp=True)
    benchmark_model(flash_model, "ErwinTransformerFlash", num_points, batch_size, use_amp=True)

csv_file.close()