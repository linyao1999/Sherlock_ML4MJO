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
# 1. Environment & Path Setup
# ======================================================================
scripts_path = Path(__file__).resolve().parents[2]
sys.path.append(str(scripts_path))
sys.path.append(str(scripts_path / 'src'))

from data_prepare.dataset import load_test_data
from models.cnnmlp import CNNMLP
from models.unet import UNet_A

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

# Directories
save_dir = Path(f"/scratch/users/linyao/ML4MJO/scripts/outputs/feature_maps/{exp_name}")
save_dir.mkdir(parents=True, exist_ok=True)

save_path = save_dir / f"spectra_{exp_name}_{trial_tag}_exp{exp_num}.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

print(f"Loading {config['model']['name']} (Ensemble Member {exp_num})...")
if config["model"]["name"] == "CNN_MLP":
    model = CNNMLP(config["model"]["cnn"], config["model"]["mlp"])
elif config["model"]["name"] == "UNet_A":
    model = UNet_A(config["model"]["cnn"], config["model"]["mlp"])

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ======================================================================
# 3. Setup Forward Hooks (Only Norm Layers Needed)
# ======================================================================
activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach().cpu().numpy()
    return hook

for name, module in model.named_modules():
    if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
        module.register_forward_hook(get_activation(name))

# ======================================================================
# 4. On-the-Fly Analysis (Spectra & Manual ReLU Extraction)
# ======================================================================
print("Loading the test dataset...")
test_loader = load_test_data(config)
total_samples = len(test_loader.dataset)

# Calculate 72 evenly spaced indices across the dataset
target_time_indices = sorted(list(set(np.linspace(0, 2100, 72).astype(int))))
start_date_obj = datetime.datetime.strptime(config["data"]["test_start"], '%Y-%m-%d')

running_spectra_sum = {}
running_weight_count = {}
example_feature_maps = {} 
relu_time_steps = {} 
actual_saved_dates = []

def process_and_accumulate_spectra(name, feature_array):
    power = np.abs(np.fft.rfft(feature_array, axis=-1)) ** 2
    power[..., 0] = 0.0 
    norm_power = power / (np.sum(power, axis=-1, keepdims=True) + 1e-12)
    
    batch_sum = np.sum(norm_power, axis=(0, 1, 2))
    batch_count = norm_power.shape[0] * norm_power.shape[1] * norm_power.shape[2]
    
    if name not in running_spectra_sum:
        running_spectra_sum[name] = batch_sum
        running_weight_count[name] = batch_count
    else:
        running_spectra_sum[name] += batch_sum
        running_weight_count[name] += batch_count

print("Running forward passes, computing spectra, and extracting manual ReLU steps...")
global_sample_idx = 0

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        print(f"  -> Processing batch {batch_idx + 1}/{len(test_loader)}", end="\r")
        
        inp_np = inputs.cpu().numpy()
        process_and_accumulate_spectra("Original Input", inp_np)
        
        inputs = inputs.to(device)
        _ = model(inputs)
        
        if batch_idx == 0:
            example_feature_maps["Original Input"] = inp_np[0:1] 
            for layer_name, fm_np in activations.items():
                example_feature_maps[layer_name] = fm_np[0:1]
        
        for layer_name, fm_np in activations.items():
            process_and_accumulate_spectra(layer_name, fm_np)
            
        batch_size = inputs.shape[0]
        for b_idx in range(batch_size):
            if global_sample_idx in target_time_indices:

                # --- NEW: Save the Original Input without ReLU ---
                if "Original Input" not in relu_time_steps:
                    relu_time_steps["Original Input"] = []
                relu_time_steps["Original Input"].append(np.mean(inp_np[b_idx], axis=0))
                # -----------------------------------------------

                for layer_name, fm_np in activations.items():
                    relu_fm = np.maximum(0, fm_np[b_idx])
                    channel_avg_map = np.mean(relu_fm, axis=0)
                    
                    if layer_name not in relu_time_steps:
                        relu_time_steps[layer_name] = []
                    relu_time_steps[layer_name].append(channel_avg_map)
                
                current_date = (start_date_obj + datetime.timedelta(days=int(global_sample_idx))).strftime('%Y-%m-%d')
                if current_date not in actual_saved_dates:
                    actual_saved_dates.append(current_date)
                    
            global_sample_idx += 1
            
        activations.clear()

print("\nFinished processing all batches. Averaging spectra...")

final_spectra = {
    name: running_spectra_sum[name] / running_weight_count[name] 
    for name in running_spectra_sum
}

# ======================================================================
# 5. Plotting Spectra (Wavenumbers 1-20)
# ======================================================================
print("Plotting spectral results...")
plt.rcParams['font.size'] = 20
fig, ax = plt.subplots(figsize=(8.5, 6.5))

layer_keys = [k for k in final_spectra.keys() if k != "Original Input"]
# cmap = plt.get_cmap('tab10')
# layer_colors = cmap.colors
layer_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink']
wavenumbers = np.arange(1, 21)

if "Original Input" in final_spectra:
    spectrum = final_spectra["Original Input"]
    ax.plot(wavenumbers, spectrum[1:21], color='black', linestyle='-', linewidth=2, label="Original Input")

for idx, key in enumerate(layer_keys):
    spectrum = final_spectra[key]
    clean_label = key.replace("layer_", "")
    color = layer_colors[idx % len(layer_colors)]
    ax.plot(wavenumbers, spectrum[1:21], color=color, linewidth=2, alpha=1.0, label=clean_label)

ax.set_xlim(0, 21)
ax.set_xticks(np.arange(0, 21, 5))
ax.set_xlabel("Zonal Wavenumber")
ax.set_ylabel("Normalized Power")
# ax.set_title(f"Zonal Power Spectra of Feature Maps\n({model_name} - Exp {exp_num})")
ax.grid(True, linestyle=':', alpha=0.6)
# ax.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.35, 1.0))
plt.tight_layout()

fig.savefig(save_path, dpi=300, bbox_inches='tight')
# np.savez(save_dir / f"computed_spectra_{exp_name}_{trial_tag}_exp{exp_num}.npz", **final_spectra)
plt.close(fig)
# ======================================================================
# 6. Plotting Selected ReLU Maps per Time Step (+ Input Layout)
# ======================================================================
print(f"Plotting individual ReLU feature maps for {len(actual_saved_dates)} time steps...")

# Filter out the input so we can safely sample just the network layers
all_relu_layers = [k for k in relu_time_steps.keys() if k != "Original Input"]

if len(all_relu_layers) > 6:
    selected_indices = np.linspace(0, len(all_relu_layers) - 1, 6).astype(int)
    selected_relu_layers = [all_relu_layers[i] for i in selected_indices]
else:
    selected_relu_layers = all_relu_layers

# Prepend the Original Input so it plots at the top
layers_to_plot = ["Original Input"] + selected_relu_layers

step_save_dir = save_dir / f"relu_steps_{exp_name}_{trial_tag}_exp{exp_num}"
step_save_dir.mkdir(parents=True, exist_ok=True)

lon = np.arange(0, 360, config["data"].get("grid_res", 2))
lat = np.arange(
    config["data"].get("lat_range", 20),
    -config["data"].get("lat_range", 20) - config["data"].get("grid_res", 2),
    -config["data"].get("grid_res", 2)
)

def set_geo_ticks(ax):
    ax.set_xticks(np.arange(30, 360, 60))
    ax.set_xticklabels(['30E', '90E', '150E', '150W', '90W', '30W'])
    ax.set_yticks(np.arange(-20, 22, 20))
    ax.set_yticklabels(['20S', '0', '20N'])
    
layer_leg = [f'{input_var_name}', 'L1','L2','L3','L4','L5','L6']
for sample_idx in range(len(actual_saved_dates)):
    current_date = actual_saved_dates[sample_idx]
    
    # Height scales automatically with the number of panels
    fig_relu, axes_relu = plt.subplots(len(layers_to_plot), 1, figsize=(10, 2.5 * len(layers_to_plot)))
    if len(layers_to_plot) == 1: axes_relu = [axes_relu] 

    for i, layer_name in enumerate(layers_to_plot):
        sel_spatial_map = relu_time_steps[layer_name][sample_idx]
        
        vmax = np.max(np.abs(sel_spatial_map))
        vmax = vmax if vmax > 0 else 1.0
        
        # Use full -1 to 1 spectrum for raw input; 0 to 1 for ReLU activations
        if layer_name == "Original Input":
            levels = np.linspace(-1, 1, 11)
            im = axes_relu[i].contourf(lon, lat, sel_spatial_map / vmax, cmap='RdBu_r', levels=levels)
        else:
            levels = np.linspace(0, 1, 11)
            im = axes_relu[i].contourf(lon, lat, sel_spatial_map / vmax, cmap='Blues', levels=levels)
            
        set_geo_ticks(axes_relu[i])
        axes_relu[i].text(5, lat[len(lat)//5], f'{layer_leg[i]}', color='black')

    plt.tight_layout()
    relu_save_path = step_save_dir / f"relu_maps_{current_date}.png"
    fig_relu.savefig(relu_save_path, dpi=300, bbox_inches='tight')
    plt.close(fig_relu)
    
print(f"Saved {len(actual_saved_dates)} daily maps to: {step_save_dir}")

# ======================================================================
# 7. Plotting Example Feature Maps (Norm Layers)
# ======================================================================
print("Plotting example feature maps...")
MAX_CHANNELS_TO_PLOT = 16 

for layer_name, feature_map in example_feature_maps.items():
    fm = feature_map[0] 
    num_channels = fm.shape[0]
    lat_size, lon_size = fm.shape[1], fm.shape[2]
    
    plot_channels = min(num_channels, MAX_CHANNELS_TO_PLOT)
    grid_size = int(np.ceil(np.sqrt(plot_channels)))
    
    fig_width = 16
    fig_height = fig_width * (2.5 / 10)  
    
    fig_ex, axes_ex = plt.subplots(grid_size, grid_size, figsize=(fig_width, fig_height))
    # fig_ex.suptitle(f"{layer_name} | Pre-ReLU Norm", fontsize=16)
    
    if isinstance(axes_ex, np.ndarray): axes_ex = axes_ex.flatten()
    else: axes_ex = [axes_ex]
    
    for i in range(len(axes_ex)):
        if i < plot_channels:
            # Pre-ReLU (Norm layer) maps will have both negative and positive values, perfectly utilizing RdBu_r
            vmax = np.max(np.abs(fm[i]))
            vmax = vmax if vmax > 0 else 1.0 
            
            axes_ex[i].contourf(lon, lat, fm[i] / vmax, cmap='RdBu_r', levels=np.linspace(-1, 1, 11))
            # axes_ex[i].set_title(f"Channel {i}", fontsize=10)
            axes_ex[i].text(5, lat[len(lat)//5], f"Channel {i}", color='black', fontsize=8)
            axes_ex[i].axis('off')
        else:
            axes_ex[i].axis('off')
            
    plt.tight_layout()
    safe_layer_name = layer_name.replace(".", "_").replace(" ", "_")
    map_save_path = save_dir / f"example_map_{exp_name}_{trial_tag}_exp{exp_num}_{safe_layer_name}.png"
    fig_ex.savefig(map_save_path, dpi=200, bbox_inches='tight')
    plt.close(fig_ex)

print("\nAll done!")
