import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime

# ======================================================================
# 1. Environment, Path Setup & Load Contributions
# ======================================================================
scripts_path = Path(__file__).resolve().parents[2]
sys.path.append(str(scripts_path))
sys.path.append(str(scripts_path / 'src'))

from data_prepare.dataset import load_test_data
from models.unet import UNet_A

# Read environment variables
dataflg = os.environ.get("dataflg", "era5").lower()
input_dir = os.environ.get("expflg", "unscaled_m10resi_wnx9resi")
input_var_name = os.environ.get("input_var", "olr")
output_var_name = os.environ.get("output_var", "RMM")
model_name = "UNet_A"
target_date_str = os.environ.get("target_date", "2016-12-20")
rangeflg = os.environ.get("rangeflg", "true").lower() == "true"
vmax = 13.0 if output_var_name == 'RMM' else 8.0
exp_name = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{input_dir}"
trial_rank = 1
trial_tag = f"t{trial_rank}"

# Find best exp_num based on BCC at lead=15 (RMM) or lead=25 (ROMI)
_target_lead = 15 if output_var_name == 'RMM' else 25
_metrics_dir = Path(f"/scratch/users/linyao/ML4MJO/scripts/outputs/metrics/{exp_name}")
_metrics_file = _metrics_dir / f"metrics_{exp_name}_{trial_tag}_test.npz"
_metrics_data = np.load(_metrics_file)
_ind_bcc = _metrics_data["ind_bcc"]
exp_num = int(np.argmax(_ind_bcc[:, _target_lead]) + 1)
print(f"Best exp_num based on BCC at lead={_target_lead}: {exp_num}")

# Directories
save_dir = Path(f"/scratch/users/linyao/ML4MJO/scripts/outputs/Figs/features/{exp_name}")
save_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Contribution Data to find TOP 10 channels OVERALL (hid2 & hid6 combined) ---
contrib_dir = Path(f"/scratch/users/linyao/ML4MJO/scripts/outputs/contribution/{exp_name}")
contrib_file = contrib_dir / f"contribution_data_{exp_name}_exp{exp_num}.npz"
loaded_data = np.load(contrib_file)

contrib_hid2 = loaded_data["hid2"]
contrib_hid6 = loaded_data["hid6"]

# Pool contributions into a list of tuples: (layer_name, channel_index, score)
all_contribs = []
for i, val in enumerate(contrib_hid2):
    all_contribs.append(("hid2", i, val))
for i, val in enumerate(contrib_hid6):
    all_contribs.append(("hid6", i, val))

# Sort descending by contribution value and grab the top 10
all_contribs.sort(key=lambda x: x[2], reverse=True)
top_10_overall = all_contribs[:10]

print("Top 10 channels overall across hid2 and hid6:")
for rank, (layer, ch, score) in enumerate(top_10_overall, 1):
    print(f"  #{rank}: {layer} - Ch {ch} (Score: {score:.4f})")

# ======================================================================
# 2. Load Config & Model
# ======================================================================
config_path = f'./yaml/best_config_{exp_name}_{trial_tag}.yaml'
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

config["data"]["test_start"] = '2016-01-01'

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

print(f"Loading UNet_A (Ensemble Member {exp_num})...")
model = UNet_A(config["model"]["cnn"], config["model"]["mlp"])
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ======================================================================
# 3. Setup Forward Hooks (hid2 and hid6 Norm Layers)
# ======================================================================
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu().numpy()
    return hook

for name, module in model.named_modules():
    if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        if "hid2" in name or "hid6" in name:
            activations_name = "hid2" if "hid2" in name else "hid6"
            module.register_forward_hook(get_activation(activations_name))

# ======================================================================
# 4. Compute target sample index from the given date
# ======================================================================
start_date_obj = datetime.datetime.strptime(config["data"]["test_start"], '%Y-%m-%d')
target_date_obj = datetime.datetime.strptime(target_date_str, '%Y-%m-%d')
target_sample_idx = (target_date_obj - start_date_obj).days
print(f"Target date: {target_date_str} -> sample index: {target_sample_idx}")

# ======================================================================
# 5. Run Forward Pass to Extract Feature Maps for the Target Date
# ======================================================================
print("Loading the test dataset...")
test_loader = load_test_data(config)

saved_input = None
saved_top10_maps = None
found = False

print("Running forward passes to find the target date...")
global_sample_idx = 0

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        batch_size = inputs.shape[0]

        # Check if target index falls within this batch
        if global_sample_idx + batch_size <= target_sample_idx:
            global_sample_idx += batch_size
            continue

        # Run forward pass for this batch
        inp_np = inputs.cpu().numpy()
        inputs = inputs.to(device)
        _ = model(inputs)

        b_idx = target_sample_idx - global_sample_idx

        # Save original input (average across channels if multi-channel)
        saved_input = np.mean(inp_np[b_idx], axis=0)

        # Extract individual top 10 channel maps after ReLU
        saved_top10_maps = []
        for layer, ch, _ in top_10_overall:
            fm = activations[layer][b_idx, ch]
            fm = np.maximum(0, fm)  # Apply ReLU
            saved_top10_maps.append(fm)

        activations.clear()
        found = True
        break

if not found:
    print(f"ERROR: target date {target_date_str} (index {target_sample_idx}) not found in test data.")
    sys.exit(1)

print(f"Successfully extracted feature maps for {target_date_str}.")

# ======================================================================
# 6. Plotting Individual Top 10 ReLU Feature Maps (single figure)
# ======================================================================
relu_tag = "relu"
print(f"Plotting individual top 10 {relu_tag} maps for {target_date_str}...")

plt.rcParams.update({'font.size': 22})

lon = np.arange(0, 360, config["data"].get("grid_res", 2))
lat = np.arange(
    config["data"].get("lat_range", 20),
    -config["data"].get("lat_range", 20) - config["data"].get("grid_res", 2),
    -config["data"].get("grid_res", 2)
)

eq_lat_indices = np.where((lat >= -10) & (lat <= 10))[0]

def set_geo_ticks(ax):
    ax.set_xticks(np.arange(30, 360, 60))
    ax.set_xticklabels(['30E', '90E', '150E', '150W', '90W', '30W'])
    ax.set_yticks(np.arange(-20, 22, 20))
    ax.set_yticklabels(['20S', '0', '20N'])

def set_lon_ticks(ax):
    ax.set_xlim(0, 360)
    ax.set_xticks(np.arange(30, 360, 60))
    ax.set_xticklabels(['30E', '90E', '150E', '150W', '90W', '30W'])
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7, color='gray')

# --- Single figure: 10 rows x 2 columns (left=map, right=line) ---
nrows = 10
fig, axes = plt.subplots(nrows, 2, figsize=(20, nrows * 3))

for rank, ((layer, ch, score), fm) in enumerate(zip(top_10_overall, saved_top10_maps)):

    # Determine vmax for this subplot
    if rangeflg:
        if output_var_name == 'ROMI':
            romi_vmax_list = [4, 8, 4, 4, 4, 4, 4, 4, 4, 4]
            cur_vmax = float(romi_vmax_list[rank])
        else:
            cur_vmax = vmax
    else:
        cur_vmax = np.ceil(fm.max() * 10) / 10 if fm.max() > 0 else 1.0

    levels = np.linspace(0, cur_vmax, 15)

    # Left column: spatial contour map
    ax_map = axes[rank, 0]
    cf = ax_map.contourf(lon, lat, fm, cmap='Blues', levels=levels, extend='max')
    set_geo_ticks(ax_map)

    # Right column: longitudinal line plot
    ax_line = axes[rank, 1]
    for lat_idx in eq_lat_indices:
        ax_line.plot(lon, fm[lat_idx, :], color='tab:blue', linewidth=1.5, alpha=1.0)
    ax_line.set_ylim(0, cur_vmax)
    if not rangeflg:
        from matplotlib.ticker import FormatStrFormatter
        ax_line.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    set_lon_ticks(ax_line)

    # Only show x-tick labels on the bottom row
    if rank < nrows - 1:
        ax_map.set_xticklabels([])
        ax_line.set_xticklabels([])

# Column titles
# axes[0, 0].set_title("Spatial Map", fontsize=14)
# axes[0, 1].set_title("Equatorial Lines (10S-10N)", fontsize=14)

fig.subplots_adjust(hspace=0.3)

# fig.suptitle(f"Top 10 Feature Maps (ReLU) — {target_date_str}", fontsize=16, y=0.995)

out_path = save_dir / f"top10_ind_{relu_tag}_{target_date_str}_{exp_name}_{trial_tag}_exp{exp_num}.png"
fig.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Saved combined figure to: {out_path}\nAll done!")
