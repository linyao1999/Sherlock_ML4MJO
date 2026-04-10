import os
import sys
import yaml
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# ======================================================================
# 1. Environment & Path Setup (Sherlock)
# ======================================================================
scripts_path = Path(__file__).resolve().parents[2]
sys.path.append(str(scripts_path))
sys.path.append(str(scripts_path / 'src'))

# Base directories on Sherlock
sherlock_base = Path("/scratch/users/linyao/ML4MJO")
yaml_dir = Path("/scratch/users/linyao/ML4MJO/scripts/exp/MD/yaml")

# Read environment variables (falling back to defaults)
dataflg = os.environ.get("dataflg", "era5").lower()
input_var_name = os.environ.get("input_var", "olr")
output_var_name = os.environ.get("output_var", "ROMI")
model_name = os.environ.get("model_name", "UNet_A")

if dataflg == 'noaa':
    eof_path = sherlock_base / "data/targets/noaa/EOFs.npz"
else:
    eof_path = sherlock_base / "data/targets/EOFs_daily1979to2012.npz"

# Evaluation parameters
# input_dirs = ['fltano120', 'unscaled_m10all_wnx1off', 'unscaled_m10all_wnx9all', 'unscaled_m10resi_wnx9resi']
input_dirs = ['unscaled_m10resi_wnx9resi']
romi_factor = 277.8929592651819  # for ERA5; std of ROMI
LEAD = 35
N_MEMBERS = 16
datesta = '2016-01-01'
dateend = '2021-12-31'

# Directory to save the final plots
save_dir = sherlock_base / "scripts/outputs/metrics/reconstructions"
save_dir.mkdir(parents=True, exist_ok=True)

# Load EOFs (indexed by day-of-year)
EOFs = np.load(eof_path)
n_eof_days = EOFs['eof1'].shape[0]

# Lead days to plot (rows in each figure)
lead_days_to_plot = list(range(0, LEAD, 5))  # [0, 5, 10, 15, 20, 25, 30]
n_rows = len(lead_days_to_plot)

# ======================================================================
# 2. Data Loading & Plotting Loop
# ======================================================================
print("\n" + "="*50)
print("=== RECONSTRUCTION PLOTS ===")
print("="*50)
def set_geo_ticks(ax):
    ax.set_xticks(np.arange(30, 360, 60))
    ax.set_xticklabels(['30E', '90E', '150E', '150W', '90W', '30W'])
    ax.set_yticks(np.arange(-20, 22, 20))
    ax.set_yticklabels(['20S', '0', '20N'])
    
plt.rcParams['font.size'] = 14

for input_dir in input_dirs:
    exp_name = f'{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{input_dir}'
    config_path = yaml_dir / f'best_config_{exp_name}_t1.yaml'

    print(f"\nProcessing experiment: {exp_name}")

    if not config_path.exists():
        print(f"  [Error] Config not found: {config_path}. Skipping.")
        continue

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    output_path = config['prediction_save_path']
    target_path = config["data"]["target_path"]

    # ------------------------------------------------------------------
    # 2a. Load ensemble predictions and compute filtered ensemble mean
    # ------------------------------------------------------------------
    fn_list = [re.sub(r"exp\d+/", f"exp{i}/", output_path) for i in range(1, N_MEMBERS + 1)]
    valid_fn_list = [fn for fn in fn_list if os.path.exists(fn)]
    if len(valid_fn_list) != N_MEMBERS:
        print(f"  [Error] Found {len(valid_fn_list)}/{N_MEMBERS} files for {input_dir}")
        sys.exit(1)

    dss = [xr.open_dataset(fn).sel(time=slice(datesta, dateend)) for fn in valid_fn_list]
    ds = xr.concat(dss, dim='exp_num')
    ds['time'] = ds.time.dt.floor("D")

    tmin = dss[0].time.min().dt.strftime("%Y-%m-%d")
    tmax = dss[0].time.max().dt.strftime("%Y-%m-%d")

    rmm = xr.open_dataarray(target_path).sel(time=slice(tmin, tmax))
    amp = (rmm[:, 0]**2 + rmm[:, 1]**2)**0.5
    ds_sel = ds.where(amp >= 1.0, drop=True)
    ds_mean = ds_sel.mean(dim='exp_num')

    # ------------------------------------------------------------------
    # 2b. Load input OLR data directly (same normalization as MapsDataset)
    # ------------------------------------------------------------------
    input_path_list = config["data"]["input_path"]
    lat_range = config["data"].get("lat_range", 20)

    input_das = []
    for ipath in input_path_list:
        da_full = xr.open_dataarray(ipath)
        da = da_full.sel(time=slice(datesta, dateend), lat=slice(lat_range, -lat_range))
        da_train = da_full.sel(time=slice('1979-01-01', '2001-12-31'), lat=slice(lat_range, -lat_range))
        input_das.append((da - da_train.mean().values) / da_train.std().values)

    if len(input_das) > 1:
        input_da = sum(input_das) / len(input_das)
    else:
        input_da = input_das[0]

    # ------------------------------------------------------------------
    # 2c. Select target dates (like a5_feature_map_top10.py)
    # ------------------------------------------------------------------
    n_times = len(ds_mean.time)
    target_time_indices = sorted(list(set(np.linspace(0, n_times-1, 72).astype(int))))

    exp_save_dir = save_dir / exp_name
    exp_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Selected {len(target_time_indices)} dates from {n_times} available.")
    print(f"  Generating figures...")

    # ------------------------------------------------------------------
    # 2d. For each date, reconstruct OLR and create a figure
    # ------------------------------------------------------------------
    for t_idx in target_time_indices:
        current_time = ds_mean.time[t_idx]
        date_str = pd.Timestamp(current_time.values).strftime('%Y-%m-%d')
        doy = pd.Timestamp(current_time.values).dayofyear - 1  # 0-indexed

        # Get prediction/truth for this date: shape (36, 2)
        pred = np.reshape(ds_mean['predictions'][t_idx].values, (36, 2))
        truth = np.reshape(ds_mean['targets'][t_idx].values, (36, 2))

        # Get input OLR for LEAD consecutive days from this date
        start_time = pd.Timestamp(current_time.values)
        end_time = start_time + pd.Timedelta(days=LEAD - 1)
        input_35day = input_da.sel(time=slice(start_time, end_time))

        if len(input_35day.time) < LEAD:
            print(f"  [Warning] Insufficient input data for {date_str} "
                  f"({len(input_35day.time)}/{LEAD} days). Skipping.")
            continue

        # Reconstruct OLR using EOFs
        if dataflg == 'noaa':
            reconstructed_olr = np.zeros((LEAD, 17, 144))
            truth_olr = np.zeros((LEAD, 17, 144))
        else:
            reconstructed_olr = np.zeros((LEAD, 21, 180))
            truth_olr = np.zeros((LEAD, 21, 180))

        for i in range(LEAD):
            eof_idx = (doy + i) % n_eof_days
            if dataflg == 'noaa':
                eof1 = np.reshape(EOFs['eof1'][eof_idx, :], (17, 144))
                eof2 = np.reshape(EOFs['eof2'][eof_idx, :], (17, 144))
            else:
                eof1 = np.reshape(EOFs['eof1'][eof_idx, :], (21, 180))
                eof2 = np.reshape(EOFs['eof2'][eof_idx, :], (21, 180))

            reconstructed_olr[i, :, :] = pred[i, 0] * eof1 + pred[i, 1] * eof2
            truth_olr[i, :, :] = truth[i, 0] * eof1 + truth[i, 1] * eof2

        # Plot figure: rows = lead days 0,5,...,30; columns = input / prediction / truth
        fig, axs = plt.subplots(n_rows, 3, figsize=(15, 2 * n_rows))

        for i, lead_day in enumerate(lead_days_to_plot):
            # Column 0: Input OLR
            axs[i, 0].contourf(
                np.arange(0, 360, 2), np.arange(20, -21, -2),
                input_35day[lead_day].values, cmap='RdBu_r',
                levels=np.linspace(-5, 5, 21)
            )
            axs[i, 0].text(300, 15, f'day {lead_day}', fontsize=14, color='black')
            set_geo_ticks(axs[i, 0])

            if dataflg == 'noaa':
                # Column 1: Reconstructed Prediction
                axs[i, 1].contourf(
                    np.arange(0, 360, 2.5), np.arange(-20, 21, 2.5),
                    reconstructed_olr[lead_day], cmap='RdBu_r',
                    levels=np.linspace(-0.15, 0.15, 21)
                )
                set_geo_ticks(axs[i, 1])

                # Column 2: Truth (EOF-reconstructed)
                axs[i, 2].contourf(
                    np.arange(0, 360, 2.5), np.arange(-20, 21, 2.5),
                    truth_olr[lead_day], cmap='RdBu_r',
                    levels=np.linspace(-0.15, 0.15, 21)
                )
                set_geo_ticks(axs[i, 2])
            else:
                # Column 1: Reconstructed Prediction
                axs[i, 1].contourf(
                    np.arange(0, 360, 2), np.arange(-20, 21, 2),
                    reconstructed_olr[lead_day], cmap='RdBu_r',
                    levels=np.linspace(-0.15, 0.15, 21)
                )
                set_geo_ticks(axs[i, 1])

                # Column 2: Truth (EOF-reconstructed)
                axs[i, 2].contourf(
                    np.arange(0, 360, 2), np.arange(-20, 21, 2),
                    truth_olr[lead_day], cmap='RdBu_r',
                    levels=np.linspace(-0.15, 0.15, 21)
                )
                set_geo_ticks(axs[i, 2])

        axs[0, 0].text(5, 15, f'OLR', fontsize=14, color='black')
        axs[0, 1].text(5, 15, f'prediction', fontsize=14, color='black')
        axs[0, 2].text(5, 15, f'truth', fontsize=14, color='black')
        
        fig.tight_layout()

        plot_path = exp_save_dir / f"reconstruction_{date_str}.png"
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"  -> Saved plots to {exp_save_dir}")

print("\nAll plotting complete!")
