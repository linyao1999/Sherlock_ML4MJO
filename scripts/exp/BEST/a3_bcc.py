import os
import sys
import yaml
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

N_ENSEMBLES = 16
TOP_TRIALS = [1,]

# Base template for prediction files
BASE_PRED_PATH = "/scratch/users/linyao/ML4MJO/scripts/outputs/predictions/{dataset_type}/{model_name}/{target_name}/lead{lead}/{exp_num}/preds_lr{lr}_bs{batch_size}_do{dropout}_cnn{channels_list_str}_k{kernel_size}_mlp{hidden_layers_str}_{optimizer}.nc"

# Directory to save the final skill metrics and plots
save_dir = Path(f"/scratch/users/linyao/ML4MJO/scripts/outputs/metrics/{exp_name}")
save_dir.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 2. Main Evaluation Loop
# ======================================================================
print("\n" + "="*50)
print("=== PART 1: CALCULATING METRICS ===")
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

    # EFFICIENCY: Build the valid file list ONLY ONCE per trial.
    fn_list = []
    for exp_num in range(1, N_ENSEMBLES + 1):
        save_kwargs["exp_num"] = f"{trial_tag}/exp{exp_num}"
        fn = BASE_PRED_PATH.format(**save_kwargs)
        if os.path.exists(fn):
            fn_list.append(fn)
        else:
            print(f"  [Error] Missing prediction file: {fn}")
            sys.exit(1)

    if not fn_list:
        print(f"[{trial_tag}] No prediction files found. Skipping.")
        continue

    # Process each period (Val and Test) using the pre-validated file list
    for period_name, dates in PERIODS.items():
        out_file = save_dir / f"metrics_{exp_name}_{trial_tag}_{period_name}.npz"
        if os.path.exists(out_file):
            continue

        print(f"\n--- Evaluating Trial {trial_rank} | Period: {period_name.upper()} ({dates['start']} to {dates['end']}) ---")
        
        ind_bcc_list = []
        ind_rmse_list = []

        # Calculate Individual Skills
        for i, fn in enumerate(fn_list):
            print(f"  -> Processing member {i+1}/{len(fn_list)} for {period_name}...")
            if multi_lead:
                bcc, rmse = mjo.get_skill_one_all_leads(
                    mjo_ind=output_var_name,
                    fn=fn,
                    datesta=dates["start"],
                    dateend=dates["end"],
                    lead_max=lead,
                    Fnmjo=target_path
                )
            else:
                bcc, rmse = mjo.get_skill_one(
                    mjo_ind=output_var_name,
                    fn=fn,
                    datesta=dates["start"],
                    dateend=dates["end"],
                    Fnmjo=target_path
                )
            ind_bcc_list.append(bcc)
            ind_rmse_list.append(rmse)

        # Calculate Ensemble Mean Skill
        print(f"  -> Computing ensemble mean for {period_name}...")
        if multi_lead:
            ens_bcc, ens_rmse = mjo.get_skill_all_leads_ensemble_mean(
                fn_list=fn_list,
                datesta=dates["start"],
                dateend=dates["end"],
                leadmjo=lead,
                Fnmjo=target_path
            )
        else:
            ens_bcc, ens_rmse = mjo.get_skill_ensemble_mean(
                fn_list=fn_list,
                datesta=dates["start"],
                dateend=dates["end"],
                leadmjo=lead,
                Fnmjo=target_path
            )
        # Save Results per period
        
        np.savez(
            out_file,
            ind_bcc=np.array(ind_bcc_list),
            ind_rmse=np.array(ind_rmse_list),
            ens_bcc=ens_bcc,
            ens_rmse=ens_rmse,
            lead=lead,
            trial_rank=trial_rank
        )
        print(f"Saved {period_name.upper()} metrics to: {out_file}")

print("\nAll evaluations complete!")


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
            data_path = save_dir / f"metrics_{exp_name}_t{trial}_{period_name}.npz"
            
            if not os.path.exists(data_path):
                print(f"  -> Missing data for Trial {trial} {period_name}. Skipping plot line.")
                continue
                
            data = np.load(data_path)
            lead_max = int(data['lead'])
            leads = np.arange(0, lead_max + 1)
            
            ens_bcc, ind_bcc = data['ens_bcc'], data['ind_bcc']
            ens_rmse, ind_rmse = data['ens_rmse'], data['ind_rmse']
            
            bcc_min, bcc_max = np.min(ind_bcc, axis=0), np.max(ind_bcc, axis=0)
            rmse_min, rmse_max = np.min(ind_rmse, axis=0), np.max(ind_rmse, axis=0)
            
            # BCC Plot Line + Shading
            ax_bcc.fill_between(leads, bcc_min, bcc_max, color=colors[i], alpha=0.15)
            ax_bcc.plot(leads, ens_bcc, '-o', color=colors[i], linewidth=2, label=labels[i])
            
            # RMSE Plot Line + Shading
            ax_rmse.fill_between(leads, rmse_min, rmse_max, color=colors[i], alpha=0.15)
            ax_rmse.plot(leads, ens_rmse, '-o', color=colors[i], linewidth=2, label=labels[i])

        # --- Formatting BCC Plot ---
        ax_bcc.axhline(y=0.5, color='black', linestyle='--', linewidth=2)
        ax_bcc.set_xticks(np.arange(0, 41, 5))
        ax_bcc.set_ylim(0.1, 1.0)
        ax_bcc.set_xlim(0, 35)
        ax_bcc.set_yticks(np.arange(0.1, 1.1, 0.2))
        ax_bcc.set_xlabel('Forecast lead (days)')
        ax_bcc.set_ylabel('BCC')
        ax_bcc.set_title(f'BCC Skill Comparison ({period_name.upper()})')
        ax_bcc.legend(loc='upper right', fontsize=14)
        ax_bcc.grid(True, linestyle=':', alpha=0.6)

        fig_bcc.tight_layout()
        bcc_save_path = save_dir / f"plot_bcc_comparison_{exp_name}_{period_name}.png"
        fig_bcc.savefig(bcc_save_path, dpi=300)

        # --- Formatting RMSE Plot ---
        ax_rmse.axhline(y=1.2, color='black', linestyle='--', linewidth=2)
        ax_rmse.axhline(y=np.sqrt(2), color='gray', linestyle='-.', linewidth=2)
        ax_rmse.set_xticks(np.arange(0, 41, 5))
        ax_rmse.set_xlim(0, 35)
        ax_rmse.set_ylim(0, 1.8)
        ax_rmse.set_xlabel('Forecast lead (days)')
        ax_rmse.set_ylabel('RMSE')
        ax_rmse.set_title(f'RMSE Skill Comparison ({period_name.upper()})')
        ax_rmse.legend(loc='upper left', fontsize=14)
        ax_rmse.grid(True, linestyle=':', alpha=0.6)

        fig_rmse.tight_layout()
        rmse_save_path = save_dir / f"plot_rmse_comparison_{exp_name}_{period_name}.png"
        fig_rmse.savefig(rmse_save_path, dpi=300)
        
        # Close figures to free memory
        plt.close(fig_bcc)
        plt.close(fig_rmse)

    print("All plots generated and saved successfully!")
