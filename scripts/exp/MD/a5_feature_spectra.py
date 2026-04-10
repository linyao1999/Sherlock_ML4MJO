import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
input_dir = os.environ.get("expflg", "unscaled_m10resi_wnx9resi")
input_var_name = os.environ.get("input_var", "olr")
output_var_name = os.environ.get("output_var", "RMM")
model_name = os.environ.get("model_name", "UNet_A")
wnx0in = os.environ.get("wnx0in", "false").lower() == "true"  #whether including wavenumber 0 into the total power; default is FALSE

exp_name = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{input_dir}"
trial_rank = 1  
trial_tag = f"t{trial_rank}"
N_ENSEMBLES = 16

# Directories
save_dir = Path(f"/scratch/users/linyao/ML4MJO/scripts/outputs/feature_maps/{exp_name}")
save_dir.mkdir(parents=True, exist_ok=True)

if wnx0in:
    agg_save_path = save_dir / f"aggregated_spectra_16exps_{exp_name}_withwnx0_{trial_tag}.npz"
    plot_save_path = save_dir / f"spectra_16exps_{exp_name}_withwnx0_{trial_tag}.png"
else:
    agg_save_path = save_dir / f"aggregated_spectra_16exps_{exp_name}_{trial_tag}.npz"
    plot_save_path = Path("/scratch/users/linyao/ML4MJO/scripts/outputs/Figs") / f"spectra_16exps_{exp_name}_{trial_tag}.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================================================================
# 2. Data Calculation & Aggregation Loop
# ======================================================================
# If the aggregated data already exists, we skip the computation entirely
if agg_save_path.exists():
    print(f"Aggregated file found! Loading data directly from:\n{agg_save_path}")
    flat_agg_data = dict(np.load(agg_save_path))
else:
    print("Aggregated file not found. Processing all 16 experiments...")
    
    # Load Config
    config_path = f'./yaml/best_config_{exp_name}_{trial_tag}.yaml'
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config["data"]["test_start"] = '2016-01-01'
    
    BASE_MODEL_PATH = "/scratch/users/linyao/ML4MJO/scripts/outputs/models/{dataset_type}/{model_name}/{target_name}/lead{lead}/{exp_num}/model_lr{lr}_bs{batch_size}_do{dropout}_cnn{channels_list_str}_k{kernel_size}_mlp{hidden_layers_str}_{optimizer}.pt"
    
    test_loader = None
    all_spectra = {} # To accumulate 16 arrays for each layer
    
    for exp_num in range(1, N_ENSEMBLES+1):
        if wnx0in:
            indiv_npz = save_dir / f"computed_spectra_{exp_name}_withwnx0_{trial_tag}_exp{exp_num}.npz"
        else:
            indiv_npz = save_dir / f"computed_spectra_{exp_name}_{trial_tag}_exp{exp_num}.npz"
        
        # Load individual experiment if already computed
        if indiv_npz.exists():
            print(f"  -> Loading Exp {exp_num} from cached disk file...")
            indiv_data = dict(np.load(indiv_npz))
            
        # Otherwise, load model and compute on-the-fly
        else:
            print(f"  -> Computing Exp {exp_num} from raw model weights...")
            
            # Lazy load the test dataset once
            if test_loader is None:
                print("     (Loading Test Dataset into Memory...)")
                test_loader = load_test_data(config)
            
            # Reconstruct kwargs for specific exp_num
            save_kwargs = {
                "dataset_type": exp_name, "lat_range": config["data"]["lat_range"], "lead": config["data"]["lead"],
                "exp_num": f"{trial_tag}/exp{exp_num}", "lr": config["training"]["learning_rate"],
                "batch_size": config["training"]["batch_size"], "dropout": config["model"]["mlp"]["dropout"],
                "channels_list_str": config["model"]["cnn"]["channels_list_str"],
                "kernel_size": config.get("kernel_size_str", f"{config['model']['cnn']['kernel_size'][0]}_{config['model']['cnn']['kernel_size'][1]}"),
                "hidden_layers_str": config.get("hidden_layers_str", "_".join(map(str, config["model"]["mlp"]["hidden_layers"]))),
                "optimizer": config["training"]["optimizer"], "model_name": config["model"]["name"],
                "target_name": config["data"]["target_vars"][0]
            }
            model_path = BASE_MODEL_PATH.format(**save_kwargs)
            
            if config["model"]["name"] == "CNN_MLP":
                model = CNNMLP(config["model"]["cnn"], config["model"]["mlp"])
            else:
                model = UNet_A(config["model"]["cnn"], config["model"]["mlp"])
                
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            # Attach Hooks
            activations = {}
            def get_activation(name):
                def hook(model, input, output):
                    activations[name] = output.detach().cpu().numpy()
                return hook
            
            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                    hooks.append(module.register_forward_hook(get_activation(name)))

            # Variables for accumulation
            running_spectra_sum = {}
            running_weight_count = {}

            def process_and_accumulate_spectra(name, feature_array):
                power = np.abs(np.fft.rfft(feature_array, axis=-1)) ** 2
                if not wnx0in:
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

            # Forward Passes
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inp_np = inputs.cpu().numpy()
                    process_and_accumulate_spectra("Original Input", inp_np)
                    
                    inputs = inputs.to(device)
                    _ = model(inputs)
                    
                    for layer_name, fm_np in activations.items():
                        process_and_accumulate_spectra(layer_name, fm_np)
                        
                    activations.clear()

            # Finalize spectra for this experiment
            indiv_data = {
                name: running_spectra_sum[name] / running_weight_count[name] 
                for name in running_spectra_sum
            }
            np.savez(indiv_npz, **indiv_data)
            
            # Clean up hooks and model memory
            for h in hooks: h.remove()
            del model
            torch.cuda.empty_cache()

        # Append to master dictionary
        for k, v in indiv_data.items():
            if k not in all_spectra: all_spectra[k] = []
            all_spectra[k].append(v)

    # ------------------------------------------------------------------
    # Aggregate (Mean, Max, Min) across the 16 arrays and save
    # ------------------------------------------------------------------
    print("\nAggregating 16 experiments...")
    flat_agg_data = {}
    for k, v_list in all_spectra.items():
        v_arr = np.array(v_list)  # Shape: (16, num_wavenumbers)
        flat_agg_data[f"{k}_mean"] = np.mean(v_arr, axis=0)
        flat_agg_data[f"{k}_max"] = np.max(v_arr, axis=0)
        flat_agg_data[f"{k}_min"] = np.min(v_arr, axis=0)
        
    np.savez(agg_save_path, **flat_agg_data)
    print(f"Saved aggregated data to: {agg_save_path}")

# ======================================================================
# 3. Plotting Spectra (Mean + Fill Between)
# ======================================================================
print("Plotting aggregated spectral results...")
plt.rcParams['font.size'] = 20
fig, ax = plt.subplots(figsize=(8.5, 6.5))

layer_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink']

if wnx0in:
    wnxstat = 0
else:
    wnxstat = 1
# wnxstat = 0
logflg = False

wavenumbers = np.arange(wnxstat, 21)
# Plot Original Input Baseline
if "Original Input_mean" in flat_agg_data:
    mean_val = flat_agg_data["Original Input_mean"]
    max_val = flat_agg_data["Original Input_max"]
    min_val = flat_agg_data["Original Input_min"]
    
    if not logflg:
        ax.plot(wavenumbers, mean_val[wnxstat:21], color='black', linestyle='-', linewidth=2)
        ax.fill_between(wavenumbers, min_val[wnxstat:21], max_val[wnxstat:21], color='black', alpha=0.1)
    else:
        ax.plot(wavenumbers, np.log10(mean_val[wnxstat:21]+1e-4), color='black', linestyle='-', linewidth=2)
        ax.fill_between(wavenumbers, np.log10(min_val[wnxstat:21]+1e-4), np.log10(max_val[wnxstat:21]+1e-4), color='black', alpha=0.1)

# Get layer keys (preserving order if possible, filtering out input & stats suffixes)
layer_keys = []
for key in flat_agg_data.keys():
    if key.endswith('_mean') and "Original Input" not in key:
        layer_keys.append(key.replace('_mean', ''))

# Plot Hidden Layers
for idx, key in enumerate(layer_keys):
    mean_val = flat_agg_data[f"{key}_mean"]
    max_val = flat_agg_data[f"{key}_max"]
    min_val = flat_agg_data[f"{key}_min"]
    
    clean_label = key.replace("layer_", "")
    color = layer_colors[idx % len(layer_colors)]
    
    if not logflg:
        # Plot Mean line
        ax.plot(wavenumbers, mean_val[wnxstat:21], color=color, linewidth=2, alpha=1.0)
        # Plot Max/Min spread
        ax.fill_between(wavenumbers, min_val[wnxstat:21], max_val[wnxstat:21], color=color, alpha=0.2)
    else:
        # Plot Mean line
        ax.plot(wavenumbers, np.log10(mean_val[wnxstat:21]+1e-4), color=color, linewidth=2, alpha=1.0)
        # Plot Max/Min spread
        ax.fill_between(wavenumbers, np.log10(min_val[wnxstat:21]+1e-4), np.log10(max_val[wnxstat:21]+1e-4), color=color, alpha=0.2)

ax.set_xlim(wnxstat-1, 21)
ax.set_xticks(np.arange(0, 21, 5))

ax.set_xlabel("Zonal Wavenumber")
ax.set_ylabel("Normalized Power")
ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

fig.savefig(plot_save_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"All done! Plot saved to:\n{plot_save_path}")
