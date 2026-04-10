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

# Read environment variables (Notice the split for unscaled vs rescaled)
dataflg = os.environ.get("dataflg", "era5").lower()
unscaled_dir = os.environ.get("unscaled_expflg", "fltano120")
rescaled_dir = os.environ.get("rescaled_expflg", "rescaled_m10all_wnx9resi")
input_var_name = os.environ.get("input_var", "olr")
output_var_name = os.environ.get("output_var", "ROMI")
model_name = os.environ.get("model_name", "UNet_A")

# Define experiment names
exp_name_unscaled = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{unscaled_dir}"
exp_name_rescaled = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_model_{unscaled_dir}_input_{rescaled_dir}"

# Define the two periods for evaluation
PERIODS = {
    # "val": {"start": "2010-01-01", "end": "2015-12-31"},
    "test": {"start": "2016-01-01", "end": "2021-12-31"}
}

N_ENSEMBLES = 16
TOP_TRIALS = [1,]

# Base template for prediction files
BASE_PRED_PATH = "/scratch/users/linyao/ML4MJO/scripts/outputs/predictions/{dataset_type}/{model_name}/{target_name}/lead{lead}/{exp_num}/preds_lr{lr}_bs{batch_size}_do{dropout}_cnn{channels_list_str}_k{kernel_size}_mlp{hidden_layers_str}_{optimizer}.nc"

# Directory to save the final skill metrics and plots (Using the RESCALED name)
save_dir = Path(f"/scratch/users/linyao/ML4MJO/scripts/outputs/metrics/{exp_name_rescaled}")
save_dir.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 2. Main Evaluation Loop
# ======================================================================
print("\n" + "="*50)
print("=== PART 1: CALCULATING METRICS (RELOADED) ===")
print("="*50)

for trial_rank in TOP_TRIALS:
    trial_tag = f"t{trial_rank}"
    
    # Load config from the original UNSCALED experiment
    config_path = f'./yaml/best_config_{exp_name_unscaled}_{trial_tag}.yaml'
    
    if not os.path.exists(config_path):
        print(f"[{trial_tag}] Config not found: {config_path}. Skipping.")
        continue
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    lead = config["data"]["lead"]
    target_path = config["data"]["target_path"]
    
    # Base kwargs for path formatting (Using the RESCALED name for the dataset_type)
    save_kwargs = {
        "dataset_type": exp_name_rescaled,
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

    # Process each period (Val and Test)
    for period_name, dates in PERIODS.items():
        out_file = save_dir / f"metrics_{exp_name_rescaled}_{trial_tag}_{period_name}.npz"
        if os.path.exists(out_file):
            print(f"Metrics already exist for {period_name.upper()}. Skipping calculation.")
            continue

        print(f"\n--- Evaluating Trial {trial_rank} | Period: {period_name.upper()} ({dates['start']} to {dates['end']}) ---")
        
        ind_bcc_list = []
        ind_rmse_list = []

        # Calculate Individual Skills
        for i, fn in enumerate(fn_list):
            print(f"  -> Processing member {i+1}/{len(fn_list)} for {period_name}...")
            bcc, rmse = mjo.get_skill_one_all_leads(
                mjo_ind=output_var_name,
                fn=fn,
                datesta=dates["start"],
                dateend=dates["end"],
                lead_max=lead,
                Fnmjo=target_path
            )
            ind_bcc_list.append(bcc)
            ind_rmse_list.append(rmse)

        # Calculate Ensemble Mean Skill
        print(f"  -> Computing ensemble mean for {period_name}...")
        ens_bcc, ens_rmse = mjo.get_skill_all_leads_ensemble_mean(
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
