import os
import sys
import yaml
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline

# Adjust paths as needed for your environment
scripts_path = Path(__file__).resolve().parents[2]
sys.path.append(str(scripts_path))
sys.path.append(str(scripts_path / 'src' / 'utils'))
import metrics as mjo

# Configuration
N_ENSEMBLES = 16
TRIAL_RANK = 1
DATE_STA = '2016-01-01'
DATE_END = '2021-12-31'
LEADS_SINGLE = [0, 5, 10, 15, 20, 25, 30, 35]
MODEL_NAME = "UNet_A"  
data_sel = 'era5'

BASE_PRED_PATH = "/scratch/users/linyao/ML4MJO/scripts/outputs/predictions/{dataset_type}/{model_name}/{target_name}/lead{lead}/t{trial_rank}/exp{exp_num}/preds_lr{lr}_bs{batch_size}_do{dropout}_cnn{channels_list_str}_k{kernel_size}_mlp{hidden_layers_str}_{optimizer}.nc"

save_dir = Path("/scratch/users/linyao/ML4MJO/scripts/outputs/Figs/BEST")
save_dir.mkdir(parents=True, exist_ok=True)

# ======================================================================
# 1. Helper: Fetch File Lists
# ======================================================================
def get_fn_list(dataflg, multi_lead, mjo_ind, lead=0):
    """Reads the YAML config and generates the list of 16 prediction files."""
    expflg = "BEST"
    exp_name = f"{dataflg}_{expflg}_{MODEL_NAME}_{mjo_ind}"
    if not multi_lead:
        exp_name += f"_lead{lead}"
        
    config_path = f'./yaml/best_config_{exp_name}_t{TRIAL_RANK}.yaml'
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    target_path = config["data"]["target_path"]
    
    kwargs = {
        "dataset_type": exp_name,
        "lead": lead if not multi_lead else config["data"]["lead"],
        "lr": config["training"]["learning_rate"],
        "batch_size": config["training"]["batch_size"],
        "dropout": config["model"]["mlp"]["dropout"],
        "channels_list_str": config["model"]["cnn"]["channels_list_str"],
        "kernel_size": config.get("kernel_size_str", f"{config['model']['cnn']['kernel_size'][0]}_{config['model']['cnn']['kernel_size'][1]}"),
        "hidden_layers_str": config.get("hidden_layers_str", "_".join(map(str, config["model"]["mlp"]["hidden_layers"]))),
        "optimizer": config["training"]["optimizer"],
        "model_name": config["model"]["name"],
        "target_name": mjo_ind,
        "trial_rank": TRIAL_RANK
    }

    fn_list = []
    for exp_num in range(1, N_ENSEMBLES + 1):
        kwargs["exp_num"] = exp_num
        fn = BASE_PRED_PATH.format(**kwargs)
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Config not found: {fn}")
        fn_list.append(fn)
        
    return fn_list, target_path

# ======================================================================
# 2. Helper: Background Plotting
# ======================================================================

def mjo_phase_background(ax, opt=None):
    if opt is not None and "axisExtent" in opt and opt["axisExtent"] > 0:
        axisExtent = opt["axisExtent"]
    else:
        axisExtent = 4

    nPhase = 8

    res = {
        # "gsnDraw": False,
        # "gsnFrame": False,
        "vpXF": 0.1,
        "vpYF": 0.8,
        "trYMinF": -axisExtent,
        "trYMaxF": axisExtent,
        "trXMinF": -axisExtent,
        "trXMaxF": axisExtent + 0.05,
        "vpWidthF": 0.45,
        "vpHeightF": 0.45,
        "tmXBFormat": "f",
        "tmYLFormat": "f",
        "tmXBLabelDeltaF": -0.75,
        "tmYLLabelDeltaF": -0.75,
        "tiXAxisFontHeightF": 0.0167,
        "tiYAxisFontHeightF": 0.0167,
        "tiDeltaF": 1.25,
        "xyLineThicknessF": 1
    }

    rad = 4. * np.arctan(1.0) / 180.
    if opt is not None and "radius" in opt and opt["radius"] > 0:
        radius = opt["radius"]
    else:
        radius = 1.0

    nCirc = 361
    theta = np.linspace(0, 360, nCirc) * rad
    xCirc = radius * np.cos(theta)
    yCirc = radius * np.sin(theta)

    if opt is not None and "tiMainString" in opt:
        res["tiMainString"] = opt["tiMainString"]


    ax.plot(xCirc, yCirc, color="black", linewidth=0.5)

    txres = {
        "fontsize": 8,
        "rotation": 90,
        "va": "center",
        "ha": "center"
    }
    # txid = ax.text(0, 0, "Phase 5 (Maritime) Phase 4", **txres)

    amres = {
        "xytext": (0.5, 0.5),
        "textcoords": "axes fraction",
        "ha": "center",
        "va": "center"
    }
    # ann3 = ax.annotate("Phase 7 (Western Pacific) Phase 6")

    plres = {
        "color": "black",
        "linewidth": 1.0,
        "linestyle": "-",
        "dashes": [8, 4]
    }
    if opt is not None and "gsLineDashPattern" in opt:
        plres["dashes"] = opt["gsLineDashPattern"]

    c45 = radius * np.cos(45 * rad)
    E = axisExtent
    R = radius

    phaLine = np.zeros((nPhase, 4))
    phaLine[0, :] = [R, E, 0, 0]
    phaLine[1, :] = [c45, E, c45, E]
    phaLine[2, :] = [0, 0, R, E]
    phaLine[3, :] = [-c45, -E, c45, E]
    phaLine[4, :] = [-R, -E, 0, 0]
    phaLine[5, :] = [-c45, -E, -c45, -E]
    phaLine[6, :] = [0, 0, -R, -E]
    phaLine[7, :] = [c45, E, -c45, -E]

    for i in range(nPhase):
        ax.plot([phaLine[i, 0], phaLine[i, 1]], [phaLine[i, 2], phaLine[i, 3]], **plres)

    # plt.show()

# ======================================================================
# 3. Main Processing & Plotting Loop
# ======================================================================
for mjo_ind in ['RMM', 'ROMI']:
    print(f"\nProcessing {mjo_ind}...")
    npz_out = save_dir / f"mean_trajectory_{mjo_ind}.npz"

    if not os.path.exists( npz_out ):
        # --- A. NOAA Multi-Lead ---
        noaa_fn_multi, noaa_truth_path = get_fn_list('noaa', multi_lead=True, mjo_ind=mjo_ind)
        DATE_END = (xr.open_dataset(noaa_fn_multi[0])['time'].max()).dt.strftime('%Y-%m-%d')
        phase_da_noaa, amp_da_noaa = mjo.get_phase_amp(mjo_ind, DATE_STA, DATE_END, noaa_truth_path)
        noaa_truth_traj, noaa_multi_ens, noaa_multi_ind = mjo.get_traj_multilead(noaa_fn_multi, phase_da_noaa, amp_da_noaa)

        # --- A. ERA5 Multi-Lead ---
        era5_fn_multi, era5_truth_path = get_fn_list('era5', multi_lead=True, mjo_ind=mjo_ind)
        phase_da_era5, amp_da_era5 = mjo.get_phase_amp(mjo_ind, DATE_STA, DATE_END, era5_truth_path)
        era5_truth_traj, era5_multi_ens, era5_multi_ind = mjo.get_traj_multilead(era5_fn_multi, phase_da_era5, amp_da_era5)

        # --- B. ERA5 Single-Lead ---
        era5_single_fns = {l: get_fn_list('era5', False, mjo_ind, l)[0] for l in LEADS_SINGLE}
        era5_single_ens = mjo.get_traj_singlelead(era5_single_fns, phase_da_era5, amp_da_era5, LEADS_SINGLE)

        # --- C. NOAA Single-Lead ---
        # Fetch NOAA's target path and calculate its own initial phase/amp
        _, noaa_truth_path = get_fn_list('noaa', False, mjo_ind, 0) 
        phase_da_noaa, amp_da_noaa = mjo.get_phase_amp(mjo_ind, DATE_STA, DATE_END, noaa_truth_path)
        noaa_single_fns = {l: get_fn_list('noaa', False, mjo_ind, l)[0] for l in LEADS_SINGLE}
        noaa_single_ens = mjo.get_traj_singlelead(noaa_single_fns, phase_da_noaa, amp_da_noaa, LEADS_SINGLE)

        # --- Save Data ---
        # Because dictionaries are saved as object arrays, loading requires allow_pickle=True
        np.savez(
            npz_out,
            era5_truth=era5_truth_traj,
            noaa_truth=noaa_truth_traj,
            era5_multi_ens=era5_multi_ens,
            noaa_multi_ens=noaa_multi_ens,
            era5_multi_ind=era5_multi_ind,
            noaa_multi_ind=noaa_multi_ind,
            era5_single_ens=era5_single_ens,
            noaa_single_ens=noaa_single_ens,
            leads_single=LEADS_SINGLE
        )
        print(f"Saved {npz_out}")
    else:
        print(f"Loading existing data from {npz_out}...")
        data = np.load(npz_out, allow_pickle=True)
        if data_sel == 'era5':
            era5_truth_traj = data['era5_truth'].item()
            era5_multi_ens = data['era5_multi_ens'].item()
            era5_multi_ind = data['era5_multi_ind'].item()
            era5_single_ens = data['era5_single_ens'].item()
        else:
            noaa_truth_traj = data['noaa_truth'].item()
            noaa_multi_ens = data['noaa_multi_ens'].item()
            noaa_multi_ind = data['noaa_multi_ind'].item()
            noaa_single_ens = data['noaa_single_ens'].item()
        
    # --- Plotting ---
    plot_groups = [[2, 5, 8], [1, 3, 6], [4, 7]]
    lead_show = 20
    plt.rcParams.update({'font.size': 28})
    for i, phases in enumerate(plot_groups):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        mjo_phase_background(ax)
        for p in phases:
            # 1. Individual Members (Multi) - Orange dots
            for member_idx in range(N_ENSEMBLES):
                if data_sel == 'era5':
                    traj_ind = era5_multi_ind[p][member_idx] 
                else:
                    traj_ind = noaa_multi_ind[p][member_idx] 
                if mjo_ind == 'RMM':
                    ax.plot(traj_ind[:lead_show, 0], traj_ind[:lead_show, 1], 'o', color='orange', markeredgecolor='none', alpha=0.5, markersize=2.0, zorder=1)
                else:
                    ax.plot(traj_ind[:lead_show, 1], -traj_ind[:lead_show, 0], 'o', color='orange', markeredgecolor='none', alpha=0.5, markersize=2.0, zorder=1)
            
            # 2. NOAA Multi Ensemble Mean - Orange solid line
            if data_sel == 'era5':
                traj_multi = era5_multi_ens[p]
            else:
                traj_multi = noaa_multi_ens[p]
            if mjo_ind == 'RMM':
                ax.plot(traj_multi[:lead_show, 0], traj_multi[:lead_show, 1], color='orange', linestyle='-', marker='o', lw=1, markersize=8, zorder=5)
                ax.plot(traj_multi[0, 0], traj_multi[0, 1], color='orange', marker='o', markeredgecolor='black', markersize=8, zorder=6)
            else:
                ax.plot(traj_multi[:lead_show, 1], -traj_multi[:lead_show, 0], color='orange', linestyle='-', marker='o', lw=1, markersize=8, zorder=5)
                ax.plot(traj_multi[0, 1], -traj_multi[0, 0], color='orange', marker='o', markeredgecolor='black', markersize=8, zorder=6)

            # 4. NOAA Single-lead - Red dashed line (SMOOTH SPLINE)
            if data_sel == 'era5':
                traj_noaa_single = era5_single_ens[p][:5]
            else:
                traj_noaa_single = noaa_single_ens[p][:5]
            t_noaa = np.arange(len(traj_noaa_single))
            t_noaa_smooth = np.linspace(0, len(traj_noaa_single) - 1, 300)
            # Interpolate x and y separately
            x_smooth_noaa = make_interp_spline(t_noaa, traj_noaa_single[:, 0], k=3)(t_noaa_smooth)
            y_smooth_noaa = make_interp_spline(t_noaa, traj_noaa_single[:, 1], k=3)(t_noaa_smooth)
            # Plot smooth dashed line, original points as dots, and starting point
            if mjo_ind == 'RMM':
                ax.plot(x_smooth_noaa, y_smooth_noaa, color='red', linestyle='-', lw=1, zorder=3)
                ax.plot(traj_noaa_single[:, 0], traj_noaa_single[:, 1], color='red', marker='o', linestyle='none', markersize=8, zorder=3)
                ax.plot(traj_noaa_single[0, 0], traj_noaa_single[0, 1], color='red', marker='o', markeredgecolor='black', markersize=8, zorder=3)
            else:
                ax.plot(y_smooth_noaa, -x_smooth_noaa, color='red', linestyle='-', lw=1, zorder=3)
                ax.plot(traj_noaa_single[:, 1], -traj_noaa_single[:, 0], color='red', marker='o', linestyle='none', markersize=8, zorder=3)
                ax.plot(traj_noaa_single[0, 1], -traj_noaa_single[0, 0], color='red', marker='o', markeredgecolor='black', markersize=8, zorder=3)

            if data_sel == 'era5':
                traj_noaa_single = noaa_single_ens[p][:5]
            else:
                traj_noaa_single = era5_single_ens[p][:5]
            t_noaa = np.arange(len(traj_noaa_single))
            t_noaa_smooth = np.linspace(0, len(traj_noaa_single) - 1, 300)
            # Interpolate x and y separately
            x_smooth_noaa = make_interp_spline(t_noaa, traj_noaa_single[:, 0], k=3)(t_noaa_smooth)
            y_smooth_noaa = make_interp_spline(t_noaa, traj_noaa_single[:, 1], k=3)(t_noaa_smooth)
            # Plot smooth dashed line, original points as dots, and starting point
            if mjo_ind == 'RMM':
                ax.plot(x_smooth_noaa, y_smooth_noaa, color='red', linestyle='--', lw=1, zorder=3)
                ax.plot(traj_noaa_single[:, 0], traj_noaa_single[:, 1], color='red', marker='o', linestyle='none', markersize=8, zorder=3)
                ax.plot(traj_noaa_single[0, 0], traj_noaa_single[0, 1], color='red', marker='o', markeredgecolor='black', markersize=8, zorder=3)
            else:
                ax.plot(y_smooth_noaa, -x_smooth_noaa, color='red', linestyle='--', lw=1, zorder=3)
                ax.plot(traj_noaa_single[:, 1], -traj_noaa_single[:, 0], color='red', marker='o', linestyle='none', markersize=8, zorder=3)
                ax.plot(traj_noaa_single[0, 1], -traj_noaa_single[0, 0], color='red', marker='o', markeredgecolor='black', markersize=8, zorder=3)

            # 5. Ground Truth - Blue solid line
            if data_sel == 'era5':
                traj_truth = era5_truth_traj[p]
            else:
                traj_truth = noaa_truth_traj[p]
            if mjo_ind == 'RMM':
                ax.plot(traj_truth[:lead_show, 0], traj_truth[:lead_show, 1], color='tab:blue', linestyle='-', marker='o', lw=1, markersize=8, zorder=2)
                ax.plot(traj_truth[0, 0], traj_truth[0, 1], color='tab:blue', marker='o', markeredgecolor='black', markersize=8, zorder=2)
            else:
                ax.plot(traj_truth[:lead_show, 1], -traj_truth[:lead_show, 0], color='tab:blue', linestyle='-', marker='o', lw=1, markersize=8, zorder=2)
                ax.plot(traj_truth[0, 1], -traj_truth[0, 0], color='tab:blue', marker='o', markeredgecolor='black', markersize=8, zorder=2)

        thrd = 1.8
        ax.set_xlim(-thrd,thrd)
        ax.set_ylim(-thrd,thrd)
        ax.set_xticks([-1,0,1])
        ax.set_yticks([-1,0,1])
        ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plot_out = save_dir / f"mean_trajectory_{mjo_ind}_{data_sel}_g{i}.pdf"
        plt.savefig(plot_out, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_out}")
        plt.close(fig) # Added to prevent memory build-up in the loop
        
print("\nAll trajectories calculated and plotted successfully!")
