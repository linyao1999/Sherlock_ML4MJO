import os
import sys
import yaml
import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import special, stats

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
# input_dir = os.environ.get("expflg", "fltano120")
input_dir = os.environ.get("expflg", "unscaled_m10resi_wnx9resi")
input_var_name = os.environ.get("input_var", "olr")
# output_var_name = os.environ.get("output_var", "ROMI")
output_var_name = os.environ.get("output_var", "RMM")
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

# Hermite projection parameters (matching scaling.py)
M_MODES = 21
C_SPEED = 51
MAX_WNX_PLOT = 20

# Directories
save_dir = Path(f"/scratch/users/linyao/ML4MJO/scripts/outputs/feature_maps/{exp_name}")
save_dir.mkdir(parents=True, exist_ok=True)

contrib_dir = Path(f"/scratch/users/linyao/ML4MJO/scripts/outputs/contribution/{exp_name}")

agg_save_path = save_dir / f"aggregated_spectra2d_16exps_{exp_name}_{trial_tag}.npz"
plot_input_save_path = Path("/scratch/users/linyao/ML4MJO/scripts/outputs/Figs/MD") / f"spectra2d_input_16exps_{exp_name}_{trial_tag}.pdf"
plot_fm_save_path = Path("/scratch/users/linyao/ML4MJO/scripts/outputs/Figs/MD") / f"spectra2d_featuremap_16exps_{exp_name}_{trial_tag}.pdf"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ======================================================================
# 2. Helper Functions
# ======================================================================
def setup_hermite_basis(lat, c=51, m=10):
    """Compute Hermite basis functions on the given latitude grid (matching scaling.py)."""
    beta = 2.28e-11
    L = np.sqrt(c / beta)
    y = lat * 110e3 / L
    dy = abs(lat[0] - lat[1]) * 110e3 / L

    phi = []
    for i in range(m):
        p = special.hermite(i)
        Hm = p(y)
        phim = np.exp(-y**2 / 2) * Hm / np.sqrt((2**i) * np.sqrt(np.pi) * math.factorial(i))
        phi.append(phim)
    phi = np.array(phi)  # Shape: (m, nlat)
    return phi, dy


def compute_2d_power(feature_maps, phi, dy):
    """
    Compute normalized 2D power spectrum in (meridional mode m, zonal wavenumber wnx).

    Parameters
    ----------
    feature_maps : ndarray, shape (batch, lat, lon)
    phi : ndarray, shape (m, lat)
    dy : float

    Returns
    -------
    normalized_power : ndarray, shape (batch, m, n_wnx)
        Power normalized by total power (wnx>0, all m).
        wnx=0 column is zeroed out.
    """
    # Meridional Hermite projection: (batch, lat, lon) x (m, lat) -> (batch, m, lon)
    coeffs = np.einsum('bjl,mj->bml', feature_maps, phi) * dy

    # Zonal FFT along longitude
    fft_coeffs = np.fft.rfft(coeffs, axis=-1)  # (batch, m, n_wnx)

    # Power spectrum
    power = np.abs(fft_coeffs) ** 2  # (batch, m, n_wnx)

    # Zero out wnx=0 and normalize by total power
    power[:, :, 0] = 0.0
    total_power = np.sum(power, axis=(1, 2), keepdims=True)  # (batch, 1, 1)
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
# 4. Load Config & Compute 2D Spectra for Each Ensemble
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

# Setup Hermite basis for the feature map latitude grid
lat_range = config["data"]["lat_range"]
nlat = config["model"]["cnn"]["nlat"]
nlon = config["model"]["cnn"]["nlon"]
lat = np.linspace(lat_range, -lat_range, nlat)
phi, dy = setup_hermite_basis(lat, c=C_SPEED, m=M_MODES)

print(f"Hermite basis: {M_MODES} modes on {nlat} lat points, nlon={nlon}")

n_wnx = nlon // 2 + 1  # number of wavenumbers from rfft

if agg_save_path.exists():
    print(f"\nAggregated file found, loading from:\n  {agg_save_path}")
    agg_data = dict(np.load(agg_save_path))
    spectra_stack = agg_data['spectra_stack']
    input_spectra_stack = agg_data['input_spectra_stack']
    ensemble_mean = agg_data['ensemble_mean']
    ensemble_std = agg_data['ensemble_std']
    t_stat = agg_data['t_stat']
    p_value = agg_data['p_value']
    input_mean = agg_data['input_mean']
else:
    print("\nAggregated file not found. Processing all 16 experiments...")
    test_loader = None
    all_ensemble_spectra = []
    all_input_spectra = []

    for exp_num in range(1, N_ENSEMBLES + 1):
        indiv_npz = save_dir / f"computed_spectra2d_{exp_name}_{trial_tag}_exp{exp_num}.npz"

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

            # Hook norm layers (network[1]) of hid2 and hid6:
            # CNN_one.network = [PeriodicConv2d, BatchNorm/GroupNorm, ReLU, Dropout]
            # Hooking network[1] captures output after norm, before ReLU.
            activations = {}

            def get_activation(name):
                def hook(module, input, output):
                    activations[name] = output.detach().cpu().numpy()
                return hook

            hooks = []
            hooks.append(model.hid2.network[1].register_forward_hook(get_activation('hid2')))
            hooks.append(model.hid6.network[1].register_forward_hook(get_activation('hid6')))

            # Accumulate spectra across batches
            running_sum = np.zeros((M_MODES, n_wnx), dtype=np.float64)
            running_count = 0
            input_running_sum = np.zeros((M_MODES, n_wnx), dtype=np.float64)
            input_running_count = 0

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    # Original input spectrum
                    inp_np = inputs.cpu().numpy()[:, 0, :, :]  # (batch, lat, lon), channel 0
                    inp_power = compute_2d_power(inp_np, phi, dy)  # (batch, m, n_wnx)
                    input_running_sum += np.sum(inp_power, axis=0)
                    input_running_count += inp_np.shape[0]

                    # Forward pass
                    inputs = inputs.to(device)
                    _ = model(inputs)

                    # Process this experiment's top channels
                    for layer_name, ch_idx in top_channels:
                        fm = activations[layer_name][:, ch_idx, :, :]  # (batch, lat, lon)
                        norm_power = compute_2d_power(fm, phi, dy)  # (batch, m, n_wnx)
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
    # Aggregate across 16 ensembles and compute significance
    # ------------------------------------------------------------------
    print("\nAggregating 16 ensemble spectra and computing significance...")
    spectra_stack = np.array(all_ensemble_spectra)    # (16, m, n_wnx)
    input_spectra_stack = np.array(all_input_spectra)  # (16, m, n_wnx)

    ensemble_mean = np.mean(spectra_stack, axis=0)
    ensemble_std = np.std(spectra_stack, axis=0, ddof=1)
    input_mean = np.mean(input_spectra_stack, axis=0)

    # One-sample t-test against uniform null hypothesis:
    # If power were uniformly distributed, each (m, wnx>0) bin
    # would have expected value = 1 / (M_MODES * (n_wnx - 1))
    n_wnx_nonzero = n_wnx - 1
    expected = np.zeros((M_MODES, n_wnx))
    expected[:, 1:] = 1.0 / (M_MODES * n_wnx_nonzero)

    t_stat, p_value = stats.ttest_1samp(spectra_stack, expected, axis=0)

    np.savez(
        agg_save_path,
        spectra_stack=spectra_stack,
        input_spectra_stack=input_spectra_stack,
        ensemble_mean=ensemble_mean,
        ensemble_std=ensemble_std,
        input_mean=input_mean,
        t_stat=t_stat,
        p_value=p_value,
    )
    print(f"Saved aggregated data to: {agg_save_path}")

# ======================================================================
# 5. Plotting
# ======================================================================
print("Plotting 2D power spectra...")
plt.rcParams['font.size'] = 22

m_range = np.arange(M_MODES)
wnx_range = np.arange(0, MAX_WNX_PLOT + 1)

# Subset for plotting (start from wnx=1)
plot_input = input_mean[:M_MODES, 1:MAX_WNX_PLOT + 1]
plot_fm = ensemble_mean[:M_MODES, 1:MAX_WNX_PLOT + 1]
wnx_plot = np.arange(1, MAX_WNX_PLOT + 1)

vmin, vmax = -2.8, -0.8
levels = np.linspace(vmin, vmax, 11)

# --- Figure 1: Original Input 2D Spectrum ---
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6.5))
log_input = np.log10(plot_input + 1e-12)
im0 = ax1.contourf(wnx_plot, m_range, log_input, levels=levels, cmap='Blues', extend='both')
fig1.colorbar(im0, ax=ax1, shrink=0.9)
ax1.set_xticks([1, 5, 10, 15, 20])
ax1.set_yticks([0, 5, 10, 15, 20])
ax1.grid(True, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
# ax1.set_aspect('equal')
plt.tight_layout()
fig1.savefig(plot_input_save_path, dpi=300, bbox_inches='tight')
plt.close(fig1)

# --- Figure 2: Feature Map 2D Spectrum (mean of 16 ensembles) ---
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6.5))
log_fm = np.log10(plot_fm + 1e-12)
im1 = ax2.contourf(wnx_plot, m_range, log_fm, levels=levels, cmap='Blues', extend='both')
fig2.colorbar(im1, ax=ax2, shrink=0.9)
ax2.set_xticks([1, 5, 10, 15, 20])
ax2.set_yticks([0, 5, 10, 15, 20])
ax2.grid(True, color='gray', linestyle=':', linewidth=0.5, alpha=0.7)
# ax2.set_aspect('equal')
plt.tight_layout()
fig2.savefig(plot_fm_save_path, dpi=300, bbox_inches='tight')
plt.close(fig2)

print(f"\nAll done! Plots saved to:\n  {plot_input_save_path}\n  {plot_fm_save_path}")
