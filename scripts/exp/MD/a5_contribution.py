import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F

# ======================================================================
# 1. Environment & Path Setup
# ======================================================================
scripts_path = Path(__file__).resolve().parents[2]
sys.path.append(str(scripts_path))
sys.path.append(str(scripts_path / 'src'))

from data_prepare.dataset import load_test_data
from models.unet import UNet_A
from models.cnnmlp import CNNMLP

# Read environment variables
dataflg = os.environ.get("dataflg", "era5").lower()
input_dir = os.environ.get("expflg", "rescaled_m10resi_wnx9resi")
input_var_name = os.environ.get("input_var", "olr")
output_var_name = os.environ.get("output_var", "ROMI")
model_name = os.environ.get("model_name", "UNet_A")
exp_num = int(os.environ.get("exp_num", 1))

exp_name = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{input_dir}"
trial_rank = 1
trial_tag = f"t{trial_rank}"

save_dir = Path(f"/scratch/users/linyao/ML4MJO/scripts/outputs/contribution/{exp_name}")
save_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================================================================
# 2. Load Config & Model
# ======================================================================
config_path = f'./yaml/best_config_{exp_name}_{trial_tag}.yaml'
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

save_kwargs = {
    "dataset_type": exp_name,
    "lat_range": config["data"]["lat_range"],
    "lead": config["data"]["lead"],
    "exp_num": f"{trial_tag}/exp{exp_num}",
    "lr": config["training"]["learning_rate"],
    "batch_size": config["training"]["batch_size"],
    "dropout": config["model"]["mlp"]["dropout"],
    "channels_list_str": config["model"]["cnn"]["channels_list_str"],
    "kernel_size": config.get("kernel_size_str", f"{config['model']['cnn']['kernel_size'][0]}_{config['model']['cnn']['kernel_size'][1]}"),
    "hidden_layers_str": config.get("hidden_layers_str", "_".join(map(str, config["model"]["mlp"]["hidden_layers"]))),
    "optimizer": config["training"]["optimizer"],
    "model_name": config["model"]["name"],
    "target_name": config["data"]["target_vars"][0]
}

BASE_MODEL_PATH = "/scratch/users/linyao/ML4MJO/scripts/outputs/models/{dataset_type}/{model_name}/{target_name}/lead{lead}/{exp_num}/model_lr{lr}_bs{batch_size}_do{dropout}_cnn{channels_list_str}_k{kernel_size}_mlp{hidden_layers_str}_{optimizer}.pt"
model_path = BASE_MODEL_PATH.format(**save_kwargs)

print(f"Loading {config['model']['name']} (Ensemble Member {exp_num})...")
if config["model"]["name"] == "CNN_MLP":
    model = CNNMLP(config["model"]["cnn"], config["model"]["mlp"])
elif config["model"]["name"] == "UNet_A":
    model = UNet_A(config["model"]["cnn"], config["model"]["mlp"])

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ======================================================================
# 3. Cache Test Data in Memory (Preserving Batches)
# ======================================================================
print("Loading and caching test dataset...")
test_loader = load_test_data(config)

cached_batches = []
total_samples = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        # Keep data in batches to prevent CUDA Out of Memory errors
        inputs = inputs.to(device)
        labels = labels.to(device)
        cached_batches.append((inputs, labels))
        total_samples += labels.shape[0]

print(f"Cached {total_samples} test samples across {len(cached_batches)} batches on {device}.")
output_size = config["model"]["mlp"]["output_size"]

# ======================================================================
# 4. Ablation Analysis Logic
# ======================================================================
def get_mse_from_cache(model, cached_data, total_samples, output_size):
    """Computes MSE over the cached test batches."""
    total_mse = 0.0
    with torch.no_grad():
        for inputs, labels in cached_data:
            preds = model(inputs)
            total_mse += F.mse_loss(preds, labels, reduction='sum').item()
            
    return total_mse / (total_samples * output_size)

print("Calculating baseline performance...")
baseline_mse = get_mse_from_cache(model, cached_batches, total_samples, output_size)
print(f"Baseline MSE: {baseline_mse:.6f}")

def evaluate_channel_ablation(model, cached_data, layer_attr, channel_idx, total_samples, output_size):
    """
    Registers a forward hook that zeros out a specific channel on the target
    layer, computes MSE over the cached test set, then removes the hook.
    """
    def zero_channel_hook(module, input, output):
        # Clone to avoid in-place modification of the computation graph
        out = output.clone()
        out[:, channel_idx, :, :] = 0.0
        return out

    target_layer = getattr(model, layer_attr)
    hook = target_layer.register_forward_hook(zero_channel_hook)
    try:
        mse = get_mse_from_cache(model, cached_data, total_samples, output_size)
    finally:
        hook.remove()  # Always remove hook even if an exception occurs
    return mse

ablation_targets = {
    "hid2": config["model"]["cnn"]["num_filters_enc"],
    "hid6": int(config["model"]["cnn"]["num_filters_enc"] * 2),
}

results = {}

for layer_name, num_channels in ablation_targets.items():
    print(f"\nAnalyzing Layer: {layer_name} ({num_channels} channels)")
    layer_diffs = []

    for c in range(num_channels):
        print(f"  -> Ablating channel {c + 1}/{num_channels}", end="\r", flush=True)
        ablated_mse = evaluate_channel_ablation(
            model, cached_batches, layer_name, c, total_samples, output_size
        )
        diff = ablated_mse - baseline_mse  # Positive = channel was useful
        layer_diffs.append(diff)

    print()  # Newline after \r progress
    results[layer_name] = np.array(layer_diffs)

# ======================================================================
# 5. Plotting
# ======================================================================
print("\nPlotting contributions...")

# Normalize both layers before concatenating and plotting
pct_hid2 = (results['hid2'] / baseline_mse) * 100
pct_hid6 = (results['hid6'] / baseline_mse) * 100
all_pct_change = np.concatenate([pct_hid2, pct_hid6])

# --- Plot 1: Ranked importance curve ---
plt.figure(figsize=(8.5, 6.5))
ranked_pct = np.sort(all_pct_change)[::-1]  # Descending
channels_rank = np.arange(len(ranked_pct))

plt.plot(channels_rank, ranked_pct, 'o-', linewidth=2)
plt.xlabel("Rank of importance")
plt.ylabel("Normalized output difference (%)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(
    save_dir / f"contribution_analysis_{exp_name}_exp{exp_num}_ranked.png",
    dpi=300
)
plt.close()

# --- Plot 2: Per-channel scatter, colored by layer ---
plt.figure(figsize=(8.5, 6.5))

idx_hid2 = np.arange(len(pct_hid2))
idx_hid6 = np.arange(len(pct_hid2), len(all_pct_change))

# Distinct colors per layer + labels for the legend
plt.plot(idx_hid2, pct_hid2, 'o', color='tab:blue', markersize=4, label='Layer: hid2')
plt.plot(idx_hid6, pct_hid6, 'o', color='tab:orange', markersize=4, label='Layer: hid6')

plt.axvline(x=len(pct_hid2) - 0.5, color='gray', linestyle='--', alpha=0.5)
plt.xlabel("Channel number")
plt.ylabel("Normalized output difference (%)")

# Add the legend
plt.legend() 

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(
    save_dir / f"contribution_analysis_{exp_name}_exp{exp_num}.png",
    dpi=300
)
plt.close()

# Save raw results (absolute MSE diffs, not %)
np.savez(save_dir / f"contribution_data_{exp_name}_exp{exp_num}.npz", **results)
print(f"\nAnalysis complete. Results saved to {save_dir}")
