import matplotlib.pyplot as plt
import numpy as np

# Load the data from the text file
file_path = "MSTopK_results.txt"
data = []
with open(file_path, "r") as file:
    for line in file:
        parts = line.split()
        batches = int(parts[0].split("=")[1])
        tensor_size = int(parts[1].split("=")[1])
        total_time = float(parts[2].split("=")[1])
        data.append((batches, tensor_size, total_time))

# Organize the data
batches_set = sorted(set(d[0] for d in data))
tensor_sizes_set = sorted(set(d[1] for d in data))
results = {b: {ts: None for ts in tensor_sizes_set} for b in batches_set}

for batch, tensor_size, total_time in data:
    results[batch][tensor_size] = total_time

# Plotting
plt.figure(figsize=(12, 8))
for batch in batches_set:
    tensor_sizes = list(results[batch].keys())
    total_times = [results[batch][ts] for ts in tensor_sizes]
    plt.plot(tensor_sizes, total_times, marker='o', label=f"Batches={batch}")

# Log-log scale for better visibility
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Tensor Size", fontsize=14)
plt.ylabel("Total Time (s)", fontsize=14)
plt.title("MSTopK", fontsize=16)
plt.legend(title="Batch Sizes", fontsize=12)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()

# Save and display the plot
output_path = "MSTopK.png"
plt.savefig(output_path)
plt.show()
