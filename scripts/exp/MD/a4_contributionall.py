import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================================
# 1. Environment & Path Setup
# ======================================================================
# Read environment variables (same as individual script)
dataflg = os.environ.get("dataflg", "era5").lower()
input_dir = os.environ.get("expflg", "unscaled_m10resi_wnx9resi")
input_var_name = os.environ.get("input_var", "olr")
output_var_name = os.environ.get("output_var", "RMM")
model_name = os.environ.get("model_name", "UNet_A")

exp_name = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{input_dir}"
save_dir = Path(f"/scratch/users/linyao/ML4MJO/scripts/outputs/contribution/{exp_name}")

num_experiments = 16

# ======================================================================
# 2. Load and Process Data
# ======================================================================
print(f"Loading ablation data for {exp_name} across {num_experiments} experiments...")

all_ranked_pct = []

for exp_num in range(1, num_experiments + 1):
    file_path = save_dir / f"contribution_data_{exp_name}_exp{exp_num}.npz"
    
    if not file_path.exists():
        print(f"  -> Warning: {file_path.name} not found. Skipping.")
        continue

    # Load the compressed arrays
    loaded_data = np.load(file_path)
    baseline_mse = loaded_data["baseline_mse"].item()

    # Extract and normalize results
    if model_name == 'CNN_MLP':
        pct_change = (loaded_data["cnn_last"] / baseline_mse) * 100
    else:
        # UNet_A: Normalize both layers before concatenating
        pct_hid2 = (loaded_data["hid2"] / baseline_mse) * 100
        pct_hid6 = (loaded_data["hid6"] / baseline_mse) * 100
        pct_change = np.concatenate([pct_hid2, pct_hid6])

    # Sort in descending order to create the "Rank of importance" curve for this experiment
    ranked_pct = np.sort(pct_change)[::-1]
    all_ranked_pct.append(ranked_pct)

if not all_ranked_pct:
    print("No data found to plot. Exiting.")
    sys.exit()

# Convert list to a 2D numpy array: Shape (N_experiments, N_channels)
all_ranked_pct = np.array(all_ranked_pct)
print(f"Successfully loaded data for {all_ranked_pct.shape[0]} experiments.")

# ======================================================================
# 3. Calculate Uncertainty (Mean, Max, Min)
# ======================================================================
mean_pct = np.mean(all_ranked_pct, axis=0)
max_pct = np.max(all_ranked_pct, axis=0)
min_pct = np.min(all_ranked_pct, axis=0)

# ======================================================================
# 4. Plotting
# ======================================================================
print("Plotting ensemble contribution analysis...")
plt.rcParams['font.size'] = 20

plt.figure(figsize=(8.5, 6.5))
channels_rank = np.arange(len(mean_pct))

# Plot the shaded min/max region
plt.fill_between(
    channels_rank, 
    min_pct, 
    max_pct, 
    color='tab:blue', 
    alpha=0.3, 
)

# Plot the mean line on top
plt.plot(
    channels_rank, 
    mean_pct, 
    'o-', 
    color='tab:blue', 
    linewidth=2, 
)

plt.xlabel("Rank of importance")
plt.ylabel("Normalized output difference (%)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

output_filename = save_dir / f"contribution_analysis_{exp_name}_all16_ranked.png"
plt.savefig(output_filename, dpi=300)
plt.close()

print(f"Plot saved successfully to: {output_filename}")
