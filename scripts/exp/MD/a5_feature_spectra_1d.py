import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ======================================================================
# 1. Environment & Path Setup
# ======================================================================
scripts_path = Path(__file__).resolve().parents[2]
sys.path.append(str(scripts_path))
sys.path.append(str(scripts_path / 'src'))

from data_prepare.dataset import load_test_data
from models.unet import UNet_A

# Read environment variables
dataflg = os.environ.get("dataflg", "era5").lower()
input_dir = os.environ.get("expflg", "fltano120")
input_var_name = os.environ.get("input_var", "olr")
output_var_name = os.environ.get("output_var", "ROMI")
model_name = os.environ.get("model_name", "UNet_A")
multi_lead = os.environ.get("multi_lead", "true").lower() == "true"

if not multi_lead:
    lead = int(os.environ.get("lead", 25))
    exp_name = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{input_dir}_lead{lead}"
else:
    exp_name = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{input_dir}"

trial_rank = 1
trial_tag = f"t{trial_rank}"
N_ENSEMBLES = 16
N_TOP_CHANNELS = 10

MAX_WNX_PLOT = 21

# Directories
save_dir = Path(f"/scratch/users/linyao/ML4MJO/scripts/outputs/feature_maps/{exp_name}")
save_dir.mkdir(parents=True, exist_ok=True)

contrib_dir = Path(f"/scratch/users/linyao/ML4MJO/scripts/outputs/contribution/{exp_name}")

agg_save_path = save_dir / f"aggregated_spectra1d_16exps_{exp_name}_{trial_tag}.npz"
plot_save_path = Path("/scratch/users/linyao/ML4MJO/scripts/outputs/Figs/MD") / f"spectra1d_16exps_{exp_name}_{trial_tag}.png"
plot_save_path.parent.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================================================================
# 2. Helper Functions
# ======================================================================
def compute_1d_power(feature_maps):
    """
    Compute normalized 1D power spectrum along longitude, averaged over latitudes.

    Parameters
    ----------
    feature_maps : ndarray, shape (batch, lat, lon)

    Returns
    -------
    normalized_power : ndarray, shape (batch, n_wnx)
        Power normalized by total power (wnx>0), averaged across latitudes.
        wnx=0 is zeroed out.
    """
    # FFT along longitude for each latitude row: (batch, lat, n_wnx)
    fft_coeffs = np.fft.rfft(feature_maps, axis=-1)

    # Power spectrum per latitude: (batch, lat, n_wnx)
    power = np.abs(fft_coeffs) ** 2

    # Average across latitudes: (batch, n_wnx)
    power = np.mean(power, axis=1)

    # Zero out wnx=0 and normalize by total power
    power[:, 0] = 0.0
    total_power = np.sum(power, axis=1, keepdims=True)  # (batch, 1)
    normalized_power = power / (total_power + 1e-12)

    return normalized_power

# ======================================================================
# 3. Helper: Find Top Channels for a Single Experiment
# ======================================================================
def get_top_channels(contrib_dir, exp_name, exp_num, n_top):
    """Load contribution data for one experiment and return its top channels."""
    contrib_file = contrib_dir / f"contribution_data_{exp_name}_exp{exp_num}.npz"
    if not contrib_file.exists():
        raise FileNotFoundError(f"Contribution file not found: {contrib_file}")

    data = np.load(contrib_file)
    hid2_contrib = data['hid2']
    hid6_contrib = data['hid6']
    n_hid2 = len(hid2_contrib)

    all_contrib = np.concatenate([hid2_contrib, hid6_contrib])
    top_indices = np.argsort(all_contrib)[::-1][:n_top]

    top_channels = []
    for idx in top_indices:
        if idx < n_hid2:
            top_channels.append(('hid2', idx))
        else:
            top_channels.append(('hid6', idx - n_hid2))
    return top_channels


# ======================================================================
# 4. Load Config & Compute 1D Spectra for Each Ensemble
# ======================================================================
config_path = f'./yaml/best_config_{exp_name}_{trial_tag}.yaml'
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
config["data"]["test_start"] = '2016-01-01'

BASE_MODEL_PATH = (
    "/scratch/users/linyao/ML4MJO/scripts/outputs/models/"
    "{dataset_type}/{model_name}/{target_name}/lead{lead}/{exp_num}/"
    "model_lr{lr}_bs{batch_size}_do{dropout}_cnn{channels_list_str}"
    "_k{kernel_size}_mlp{hidden_layers_str}_{optimizer}.pt"
)

nlon = config["model"]["cnn"]["nlon"]
n_wnx = nlon // 2 + 1  # number of wavenumbers from rfft
print(f"nlon={nlon}, n_wnx={n_wnx}")

if agg_save_path.exists():
    print(f"\nAggregated file found, loading from:\n  {agg_save_path}")
    agg_data = dict(np.load(agg_save_path))
    spectra_stack = agg_data['spectra_stack']
    input_spectra_stack = agg_data['input_spectra_stack']
    ensemble_mean = agg_data['ensemble_mean']
    ensemble_std = agg_data['ensemble_std']
    input_mean = agg_data['input_mean']
else:
    print("\nAggregated file not found. Processing all 16 experiments...")
    test_loader = None
    all_ensemble_spectra = []
    all_input_spectra = []

    for exp_num in range(1, N_ENSEMBLES + 1):
        indiv_npz = save_dir / f"computed_spectra1d_{exp_name}_{trial_tag}_exp{exp_num}.npz"

        if indiv_npz.exists():
            print(f"  -> Loading Exp {exp_num} from cached file...")
            cached = np.load(indiv_npz)
            mean_spectrum = cached['mean_spectrum']
            input_spectrum = cached['input_spectrum']

        else:
            print(f"  -> Computing Exp {exp_num}...")

            # Find top channels for THIS experiment from its own contribution data
            top_channels = get_top_channels(contrib_dir, exp_name, exp_num, N_TOP_CHANNELS)
            print(f"     Top {N_TOP_CHANNELS} channels for exp {exp_num}:")
            for layer, ch in top_channels:
                print(f"       {layer} ch{ch}")

            # Lazy load test data once
            if test_loader is None:
                print("     (Loading Test Dataset...)")
                test_loader = load_test_data(config)

            # Load model
            save_kwargs = {
                "dataset_type": exp_name,
                "lat_range": config["data"]["lat_range"],
                "lead": config["data"]["lead"],
                "exp_num": f"{trial_tag}/exp{exp_num}",
                "lr": config["training"]["learning_rate"],
                "batch_size": config["training"]["batch_size"],
                "dropout": config["model"]["mlp"]["dropout"],
                "channels_list_str": config["model"]["cnn"]["channels_list_str"],
                "kernel_size": config.get(
                    "kernel_size_str",
                    f"{config['model']['cnn']['kernel_size'][0]}_{config['model']['cnn']['kernel_size'][1]}"
                ),
                "hidden_layers_str": config.get(
                    "hidden_layers_str",
                    "_".join(map(str, config["model"]["mlp"]["hidden_layers"]))
                ),
                "optimizer": config["training"]["optimizer"],
                "model_name": config["model"]["name"],
                "target_name": config["data"]["target_vars"][0]
            }
            model_path = BASE_MODEL_PATH.format(**save_kwargs)

            model = UNet_A(config["model"]["cnn"], config["model"]["mlp"])
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            # Hook norm layers (network[1]) of hid2 and hid6
            activations = {}

            def get_activation(name):
                def hook(module, input, output):
                    activations[name] = output.detach().cpu().numpy()
                return hook

            hooks = []
            hooks.append(model.hid2.network[1].register_forward_hook(get_activation('hid2')))
            hooks.append(model.hid6.network[1].register_forward_hook(get_activation('hid6')))

            # Accumulate spectra across batches
            running_sum = np.zeros(n_wnx, dtype=np.float64)
            running_count = 0
            input_running_sum = np.zeros(n_wnx, dtype=np.float64)
            input_running_count = 0

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    # Original input spectrum
                    inp_np = inputs.cpu().numpy()[:, 0, :, :]  # (batch, lat, lon), channel 0
                    inp_power = compute_1d_power(inp_np)  # (batch, n_wnx)
                    input_running_sum += np.sum(inp_power, axis=0)
                    input_running_count += inp_np.shape[0]

                    # Forward pass
                    inputs = inputs.to(device)
                    _ = model(inputs)

                    # Process this experiment's top channels
                    for layer_name, ch_idx in top_channels:
                        fm = activations[layer_name][:, ch_idx, :, :]  # (batch, lat, lon)
                        norm_power = compute_1d_power(fm)  # (batch, n_wnx)
                        running_sum += np.sum(norm_power, axis=0)
                        running_count += fm.shape[0]

                    activations.clear()

            # Average across time and feature maps
            mean_spectrum = running_sum / running_count
            input_spectrum = input_running_sum / input_running_count

            np.savez(indiv_npz, mean_spectrum=mean_spectrum, input_spectrum=input_spectrum)

            for h in hooks:
                h.remove()
            del model
            torch.cuda.empty_cache()

        all_ensemble_spectra.append(mean_spectrum)
        all_input_spectra.append(input_spectrum)

    # ------------------------------------------------------------------
    # Aggregate across 16 ensembles
    # ------------------------------------------------------------------
    print("\nAggregating 16 ensemble spectra...")
    spectra_stack = np.array(all_ensemble_spectra)    # (16, n_wnx)
    input_spectra_stack = np.array(all_input_spectra)  # (16, n_wnx)

    ensemble_mean = np.mean(spectra_stack, axis=0)
    ensemble_std = np.std(spectra_stack, axis=0, ddof=1)
    input_mean = np.mean(input_spectra_stack, axis=0)

    np.savez(
        agg_save_path,
        spectra_stack=spectra_stack,
        input_spectra_stack=input_spectra_stack,
        ensemble_mean=ensemble_mean,
        ensemble_std=ensemble_std,
        input_mean=input_mean,
    )
    print(f"Saved aggregated data to: {agg_save_path}")

# ======================================================================
# 5. Plotting
# ======================================================================
print("Plotting 1D power spectra...")
plt.rcParams['font.size'] = 22

wnx_plot = np.arange(1, MAX_WNX_PLOT + 1)
plot_input = input_mean[1:MAX_WNX_PLOT + 1]
plot_fm = ensemble_mean[1:MAX_WNX_PLOT + 1]

fig, ax = plt.subplots(1, 1, figsize=(8, 6.5))
ax.plot(wnx_plot, plot_input, color='tab:blue', linewidth=2, label='Input')
ax.plot(wnx_plot, plot_fm, color='tab:orange', linewidth=2, label='Feature map')
ax.set_xlabel("Zonal wavenumber")
ax.set_ylabel("Normalized power")
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_ylim([0, 0.18])
ax.set_yticks(np.arange(0,0.18,0.04))
ax.legend()
ax.grid(True, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
plt.tight_layout()
fig.savefig(plot_save_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"\nAll done! Plot saved to:\n  {plot_save_path}")
