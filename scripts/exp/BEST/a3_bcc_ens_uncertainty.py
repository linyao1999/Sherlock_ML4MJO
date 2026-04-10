import os
import sys
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================================
# 1. Environment & Path Setup
# ======================================================================
scripts_path = Path(__file__).resolve().parents[2]
sys.path.append(str(scripts_path))
sys.path.append(str(scripts_path / 'src'))

# Add utils path and import metrics
src_utils_path = scripts_path / "src" / "utils"
sys.path.append(str(src_utils_path))
import metrics as mjo

# Read environment variables
dataflg = os.environ.get("dataflg", "era5").lower()
expflg = os.environ.get("expflg", "BEST")
output_var_name = os.environ.get("output_var", "ROMI")
model_name = os.environ.get("model_name", "UNet_A")
multi_lead = os.environ.get("multi_lead", "true").lower() == "true"

if not multi_lead:
    lead = int(os.environ.get("lead", 0))
    exp_name = f"{dataflg}_{expflg}_{model_name}_{output_var_name}_lead{lead}"
else:
    exp_name = f"{dataflg}_{expflg}_{model_name}_{output_var_name}"

# Define the two periods for evaluation
PERIODS = {
    "val": {"start": "2010-01-01", "end": "2015-12-31"},
    "test": {"start": "2016-01-01", "end": "2021-12-31"}
}

N_TOTAL_ENSEMBLES = 16
N_SAMPLE_SIZE = 10
N_ITERATIONS = 20
TOP_TRIALS = [1,]

# Base template for prediction files
BASE_PRED_PATH = "/scratch/users/linyao/ML4MJO/scripts/outputs/predictions/{dataset_type}/{model_name}/{target_name}/lead{lead}/{exp_num}/preds_lr{lr}_bs{batch_size}_do{dropout}_cnn{channels_list_str}_k{kernel_size}_mlp{hidden_layers_str}_{optimizer}.nc"

# Directory to save the final skill metrics and plots
save_dir = Path(f"/scratch/users/linyao/ML4MJO/scripts/outputs/metrics/{exp_name}")
save_dir.mkdir(parents=True, exist_ok=True)

# Dictionary to hold our calculated metrics in memory for plotting
all_results = {}

# ======================================================================
# 2. Main Evaluation Loop
# ======================================================================
print("\n" + "="*50)
print(f"=== PART 1: CALCULATING SUBSAMPLED ENSEMBLES ({N_ITERATIONS} iterations of {N_SAMPLE_SIZE}) ===")
print("="*50)

for trial_rank in TOP_TRIALS:
    trial_tag = f"t{trial_rank}"
    config_path = f'./yaml/best_config_{exp_name}_{trial_tag}.yaml'
    
    if not os.path.exists(config_path):
        print(f"[{trial_tag}] Config not found: {config_path}. Skipping.")
        continue
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    lead = config["data"]["lead"]
    target_path = config["data"]["target_path"]
    
    # Base kwargs for path formatting
    save_kwargs = {
        "dataset_type": exp_name,
        "lat_range": config["data"]["lat_range"],
        "lead": lead,
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

    # Build the valid file list ONLY ONCE per trial.
    fn_list = []
    for exp_num in range(1, N_TOTAL_ENSEMBLES + 1):
        save_kwargs["exp_num"] = f"{trial_tag}/exp{exp_num}"
        fn = BASE_PRED_PATH.format(**save_kwargs)
        if os.path.exists(fn):
            fn_list.append(fn)
        else:
            print(f"  [Error] Missing prediction file: {fn}")
            sys.exit(1)

    if len(fn_list) < N_SAMPLE_SIZE:
        print(f"[{trial_tag}] Not enough files to sample {N_SAMPLE_SIZE}. Found {len(fn_list)}. Skipping.")
        continue

    all_results[trial_rank] = {}

    # Process each period (Val and Test)
    for period_name, dates in PERIODS.items():
        print(f"\n--- Evaluating Trial {trial_rank} | Period: {period_name.upper()} ({dates['start']} to {dates['end']}) ---")
        
        iter_bcc_list = []
        iter_rmse_list = []

        # Bootstrap: Choose 10 random members, 20 times
        for iter_idx in range(N_ITERATIONS):
            print(f"  -> Subsampling iteration {iter_idx+1}/{N_ITERATIONS}...")
            
            # Randomly select 10 unique files without replacement
            sampled_fns = random.sample(fn_list, N_SAMPLE_SIZE)
            
            if multi_lead:
                ens_bcc, ens_rmse = mjo.get_skill_all_leads_ensemble_mean(
                    fn_list=sampled_fns,
                    datesta=dates["start"],
                    dateend=dates["end"],
                    leadmjo=lead,
                    Fnmjo=target_path
                )
            else:
                ens_bcc, ens_rmse = mjo.get_skill_ensemble_mean(
                    fn_list=sampled_fns,
                    datesta=dates["start"],
                    dateend=dates["end"],
                    leadmjo=lead,
                    Fnmjo=target_path
                )
            
            iter_bcc_list.append(ens_bcc)
            iter_rmse_list.append(ens_rmse)

        # Store in memory for plotting
        all_results[trial_rank][period_name] = {
            'bcc': np.array(iter_bcc_list),
            'rmse': np.array(iter_rmse_list),
            'lead': lead
        }

print("\nAll evaluations complete! No .npz files were saved.")


if multi_lead:
    # ======================================================================
    # 3. Plotting
    # ======================================================================
    print("\n" + "="*50)
    print("=== PART 2: PLOTTING ===")
    print("="*50)

    plt.rcParams['font.size'] = 20
    colors = ['blue', 'red', 'green']
    labels = ['Trial 1', 'Trial 2', 'Trial 3']

    # Generate separate plots for 'val' and 'test'
    for period_name in PERIODS.keys():
        print(f"Generating plots for: {period_name.upper()}...")
        
        fig_bcc, ax_bcc = plt.subplots(1, 1, figsize=(8.5, 6.5))
        fig_rmse, ax_rmse = plt.subplots(1, 1, figsize=(8.5, 6.5))

        for i, trial in enumerate(TOP_TRIALS):
            if trial not in all_results or period_name not in all_results[trial]:
                print(f"  -> Missing data for Trial {trial} {period_name}. Skipping plot line.")
                continue
                
            data = all_results[trial][period_name]
            lead_max = int(data['lead'])
            leads = np.arange(0, lead_max + 1)
            
            # Arrays contain the 20 iterations: shape (20, leads)
            bcc_all = data['bcc']
            rmse_all = data['rmse']
            
            # Calculate the mean and bounds of the 20 subsamples
            bcc_mean = np.mean(bcc_all, axis=0)
            bcc_min, bcc_max = np.min(bcc_all, axis=0), np.max(bcc_all, axis=0)
            
            rmse_mean = np.mean(rmse_all, axis=0)
            rmse_min, rmse_max = np.min(rmse_all, axis=0), np.max(rmse_all, axis=0)
            
            # BCC Plot Line + Shading
            ax_bcc.fill_between(leads, bcc_min, bcc_max, color=colors[i], alpha=0.15)
            ax_bcc.plot(leads, bcc_mean, '-o', color=colors[i], linewidth=2, label=labels[i])
            
            # RMSE Plot Line + Shading
            ax_rmse.fill_between(leads, rmse_min, rmse_max, color=colors[i], alpha=0.15)
            ax_rmse.plot(leads, rmse_mean, '-o', color=colors[i], linewidth=2, label=labels[i])

        # --- Formatting BCC Plot ---
        ax_bcc.axhline(y=0.5, color='black', linestyle='--', linewidth=2)
        ax_bcc.set_xticks(np.arange(0, 41, 5))
        ax_bcc.set_ylim(0.1, 1.0)
        ax_bcc.set_xlim(0, 35)
        ax_bcc.set_yticks(np.arange(0.1, 1.1, 0.2))
        ax_bcc.set_xlabel('Forecast lead (days)')
        ax_bcc.set_ylabel('BCC')
        ax_bcc.set_title(f'BCC Subsampled ({N_SAMPLE_SIZE} members) - {period_name.upper()}')
        ax_bcc.legend(loc='upper right', fontsize=14)
        ax_bcc.grid(True, linestyle=':', alpha=0.6)

        fig_bcc.tight_layout()
        bcc_save_path = save_dir / f"plot_bcc_subsample_{N_SAMPLE_SIZE}x{N_ITERATIONS}_{exp_name}_{period_name}.png"
        fig_bcc.savefig(bcc_save_path, dpi=300)

        # --- Formatting RMSE Plot ---
        ax_rmse.axhline(y=1.2, color='black', linestyle='--', linewidth=2)
        ax_rmse.axhline(y=np.sqrt(2), color='gray', linestyle='-.', linewidth=2)
        ax_rmse.set_xticks(np.arange(0, 41, 5))
        ax_rmse.set_xlim(0, 35)
        ax_rmse.set_ylim(0, 1.8)
        ax_rmse.set_xlabel('Forecast lead (days)')
        ax_rmse.set_ylabel('RMSE')
        ax_rmse.set_title(f'RMSE Subsampled ({N_SAMPLE_SIZE} members) - {period_name.upper()}')
        ax_rmse.legend(loc='upper left', fontsize=14)
        ax_rmse.grid(True, linestyle=':', alpha=0.6)

        fig_rmse.tight_layout()
        rmse_save_path = save_dir / f"plot_rmse_subsample_{N_SAMPLE_SIZE}x{N_ITERATIONS}_{exp_name}_{period_name}.png"
        fig_rmse.savefig(rmse_save_path, dpi=300)
        
        plt.close(fig_bcc)
        plt.close(fig_rmse)

    print("All plots generated and saved successfully!")
