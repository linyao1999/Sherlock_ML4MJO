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
input_dir = os.environ.get("expflg", "fltano120")
# input_dir = os.environ.get("expflg", "unscaled_m10resi_wnx9resi")
input_var_name = os.environ.get("input_var", "olr")
output_var_name = os.environ.get("output_var", "ROMI")
model_name = "UNet_A"
before_relu = os.environ.get("before_relu", "true").lower() == "true"
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
save_dir = Path(f"/scratch/users/linyao/ML4MJO/scripts/outputs/feature_maps/{exp_name}")
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
# 4. On-the-Fly Analysis (Extracting Mean of Top 10 ReLU Maps)
# ======================================================================
print("Loading the test dataset...")
test_loader = load_test_data(config)

target_time_indices = sorted(list(set(np.linspace(0, 2100, 72).astype(int))))
start_date_obj = datetime.datetime.strptime(config["data"]["test_start"], '%Y-%m-%d')

saved_inputs = []
saved_mean_top10 = []
actual_saved_dates = []

print("Running forward passes and extracting target feature maps...")
global_sample_idx = 0

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        print(f"  -> Processing batch {batch_idx + 1}/{len(test_loader)}", end="\r")
        
        inp_np = inputs.cpu().numpy()
        inputs = inputs.to(device)
        _ = model(inputs)
        
        batch_size = inputs.shape[0]
        for b_idx in range(batch_size):
            if global_sample_idx in target_time_indices:
                
                # 1. Save original input (average across channels if multi-channel)
                saved_inputs.append(np.mean(inp_np[b_idx], axis=0))

                # 2. Extract top 10 channels and calculate the mean spatial map
                top10_spatial_maps = []
                for layer, ch, _ in top_10_overall:
                    fm = activations[layer][b_idx, ch]
                    if not before_relu:
                        fm = np.maximum(0, fm)
                    top10_spatial_maps.append(fm)
                
                mean_top10_map = np.mean(top10_spatial_maps, axis=0)
                saved_mean_top10.append(mean_top10_map)
                
                current_date = (start_date_obj + datetime.timedelta(days=int(global_sample_idx))).strftime('%Y-%m-%d')
                if current_date not in actual_saved_dates:
                    actual_saved_dates.append(current_date)
                    
            global_sample_idx += 1
            
        activations.clear()

print("\nFinished processing all batches.")
# ======================================================================
# 5. Plotting Selected ReLU Maps & Equatorial Line Plots
# ======================================================================
relu_tag = "pre_relu" if before_relu else "relu"
print(f"Plotting Input & Mean Top 10 {relu_tag} maps for {len(actual_saved_dates)} time steps...")

# Apply global font size
plt.rcParams.update({'font.size': 24})

step_save_dir = save_dir / f"{relu_tag}_steps_{exp_name}_{trial_tag}_exp{exp_num}"
step_save_dir.mkdir(parents=True, exist_ok=True)

lon = np.arange(0, 360, config["data"].get("grid_res", 2))
lat = np.arange(
    config["data"].get("lat_range", 20),
    -config["data"].get("lat_range", 20) - config["data"].get("grid_res", 2),
    -config["data"].get("grid_res", 2)
)

# Identify indices for latitudes between -10 and 10 degrees for the line plots
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

# --- Define Absolute Layout Boxes ---
# format: [left, bottom, width, height] as fractions of the figure size
# This guarantees the main plots are identically sized across all figures
MAIN_AX_BOX = [0.1, 0.25, 0.72, 0.7]
CBAR_AX_BOX = [0.84, 0.25, 0.02, 0.7]
STANDARD_FIGSIZE = (14, 4)

for sample_idx, current_date in enumerate(actual_saved_dates):
    
    inp_map = saved_inputs[sample_idx]
    mean_map = saved_mean_top10[sample_idx]

    # ------------------------------------------------------------------
    # Plot 1a: Original Input (Spatial Contour)
    # ------------------------------------------------------------------
    fig_inp = plt.figure(figsize=STANDARD_FIGSIZE)
    ax_inp = fig_inp.add_axes(MAIN_AX_BOX)
    cax_inp = fig_inp.add_axes(CBAR_AX_BOX) # Explicit colorbar axis
    
    levels_inp = np.linspace(-6, 6, 13) 
    
    cf_inp = ax_inp.contourf(lon, lat, inp_map, cmap='RdBu_r', levels=levels_inp, extend='both')
    set_geo_ticks(ax_inp)
    
    fig_inp.colorbar(cf_inp, cax=cax_inp, format='%.1f')
    
    # NO plt.tight_layout() here to protect explicit dimensions
    fig_inp.savefig(step_save_dir / f"input_{current_date}.png", dpi=300, bbox_inches='tight')
    plt.close(fig_inp)

    # ------------------------------------------------------------------
    # Plot 1b: Original Input (Longitudinal Line Plots for -10 to 10 lat)
    # ------------------------------------------------------------------
    fig_inp_line = plt.figure(figsize=STANDARD_FIGSIZE)
    ax_inp_line = fig_inp_line.add_axes(MAIN_AX_BOX) # Same main axis box
    
    for lat_idx in eq_lat_indices:
        ax_inp_line.plot(lon, inp_map[lat_idx, :], color='tab:blue', linewidth=2, alpha=1.0)
        
    ax_inp_line.set_ylim(-6, 6)
    set_lon_ticks(ax_inp_line)
    
    fig_inp_line.savefig(step_save_dir / f"input_lines_{current_date}.png", dpi=300, bbox_inches='tight')
    plt.close(fig_inp_line)

    # ------------------------------------------------------------------
    # Plot 2a: Mean of Top 10 ReLU Feature Maps (Spatial Contour)
    # ------------------------------------------------------------------
    fig_mean = plt.figure(figsize=STANDARD_FIGSIZE)
    ax_mean = fig_mean.add_axes(MAIN_AX_BOX)
    cax_mean = fig_mean.add_axes(CBAR_AX_BOX) # Explicit colorbar axis
    
    if before_relu:
        if output_var_name == 'RMM':
            vmax_mean = 2.0
        else:
            vmax_mean = 2.0
        levels_mean = np.linspace(-vmax_mean, vmax_mean, 11)
        cf_mean = ax_mean.contourf(lon, lat, mean_map, cmap='RdBu_r', levels=levels_mean, extend='both')
    else:
        if output_var_name == 'RMM':
            levels_mean = np.linspace(0, 2.6, 11)
        else:
            levels_mean = np.linspace(0, 1.0, 11)
        cf_mean = ax_mean.contourf(lon, lat, mean_map, cmap='Blues', levels=levels_mean, extend='max')
    set_geo_ticks(ax_mean)
    
    fig_mean.colorbar(cf_mean, cax=cax_mean, format='%.1f')
    
    fig_mean.savefig(step_save_dir / f"mean_top10_{relu_tag}_{current_date}.png", dpi=300, bbox_inches='tight')
    plt.close(fig_mean)

    # ------------------------------------------------------------------
    # Plot 2b: Mean of Top 10 ReLU (Longitudinal Line Plots for -10 to 10 lat)
    # ------------------------------------------------------------------
    fig_mean_line = plt.figure(figsize=STANDARD_FIGSIZE)
    ax_mean_line = fig_mean_line.add_axes(MAIN_AX_BOX) # Same main axis box
    
    for lat_idx in eq_lat_indices:
        ax_mean_line.plot(lon, mean_map[lat_idx, :], color='tab:blue', linewidth=2, alpha=1.0)
        
    if before_relu:
        if output_var_name == 'RMM':
            ax_mean_line.set_ylim(-2.6, 2.6)
        else:
            ax_mean_line.set_ylim(-1.0, 1.0)
            ax_mean_line.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    else:
        if output_var_name == 'RMM':
            ax_mean_line.set_ylim(0, 2.6)
        else:
            ax_mean_line.set_ylim(0, 1.0)
            ax_mean_line.set_yticks([0, 0.5, 1.0])

    set_lon_ticks(ax_mean_line)

    fig_mean_line.savefig(step_save_dir / f"mean_top10_{relu_tag}_lines_{current_date}.png", dpi=300, bbox_inches='tight')
    plt.close(fig_mean_line)
    
print(f"Saved separate daily maps and line distributions to: {step_save_dir}\nAll done!")
