import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================================
# 1. Configuration & Setup
# ======================================================================
# Fixed parameters for reconstructing the exp_name
dataflg = os.environ.get("dataflg", "era5").lower()
input_var_name = os.environ.get("input_var", "olr")
output_var_name = os.environ.get("output_var", "ROMI")
model_name = os.environ.get("model_name", "UNet_A")

# Assuming we are plotting Trial 1 based on your previous TOP_TRIALS = [1,]
trial_tag = "t1" 
PERIODS = ["test"]

# Define the two groups of experiments
groups = {
    "Rescaled_Group": [
        'fltano120', 
        'rescaled_m10all_wnx1off', 
        'rescaled_m10all_wnx9all', 
        'rescaled_m10resi_wnx9resi'
    ],
    "Unscaled_Group": [
        'fltano120', 
        'unscaled_m10all_wnx1off', 
        'unscaled_m10all_wnx9all', 
        'unscaled_m10resi_wnx9resi'
    ]
}

# The colors you requested ('gold' is used for 'yellow' to ensure visibility)
COLORS = ['green', 'gold', 'orange', 'red']

# Base directory where your previous script saved the metrics
metrics_base_dir = Path("/scratch/users/linyao/ML4MJO/scripts/outputs/metrics")

# Directory to save these new combined comparison plots
save_dir = metrics_base_dir / "group_comparisons"
save_dir.mkdir(parents=True, exist_ok=True)

# ======================================================================
# 2. Plotting Loop
# ======================================================================
print("\n" + "="*50)
print("=== GENERATING MULTI-EXPERIMENT PLOTS ===")
print("="*50)

# Slightly smaller font to accommodate the longer legend labels
plt.rcParams['font.size'] = 20 

for group_name, expflg_list in groups.items():
    print(f"\nProcessing {group_name}...")
    
    for period_name in PERIODS:
        print(f"  -> Generating plots for period: {period_name.upper()}")
        
        fig_bcc, ax_bcc = plt.subplots(1, 1, figsize=(8.5, 6.5))
        fig_rmse, ax_rmse = plt.subplots(1, 1, figsize=(8.5, 6.5))
        
        for i, expflg in enumerate(expflg_list):
            # Construct the exact exp_name to locate the directory and file
            exp_name = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{expflg}"
            data_path = metrics_base_dir / exp_name / f"metrics_{exp_name}_{trial_tag}_{period_name}.npz"
            
            if not data_path.exists():
                print(f"     [Warning] Missing data for {exp_name}. Skipping line.")
                continue
                
            # Load the pre-calculated metrics
            data = np.load(data_path)
            lead_max = int(data['lead'])
            leads = np.arange(0, lead_max + 1)
            
            ens_bcc, ind_bcc = data['ens_bcc'], data['ind_bcc']
            ens_rmse, ind_rmse = data['ens_rmse'], data['ind_rmse']
            
            # Calculate ensemble spread / uncertainty
            bcc_min, bcc_max = np.min(ind_bcc, axis=0), np.max(ind_bcc, axis=0)
            rmse_min, rmse_max = np.min(ind_rmse, axis=0), np.max(ind_rmse, axis=0)
            
            # Plot BCC (Shading + Ensemble Mean)
            ax_bcc.fill_between(leads, bcc_min, bcc_max, color=COLORS[i], alpha=0.15)
            ax_bcc.plot(leads, ens_bcc, '-o', color=COLORS[i], linewidth=2)
            
            # Plot RMSE (Shading + Ensemble Mean)
            ax_rmse.fill_between(leads, rmse_min, rmse_max, color=COLORS[i], alpha=0.15)
            ax_rmse.plot(leads, ens_rmse, '-o', color=COLORS[i], linewidth=2)

        # --- Formatting BCC Plot ---
        ax_bcc.axhline(y=0.5, color='black', linestyle='--', linewidth=2)
        ax_bcc.set_xticks(np.arange(0, 41, 5))
        ax_bcc.set_xlim(0, 35)
        ax_bcc.set_ylim(0.1, 1.0)
        ax_bcc.set_yticks(np.arange(0.1, 1.1, 0.2))
        ax_bcc.set_xlabel('Forecast lead (days)')
        ax_bcc.set_ylabel('BCC')
        # ax_bcc.set_title(f'BCC Skill Comparison: {group_name.replace("_", " ")} ({period_name.upper()})')
        # ax_bcc.legend(loc='upper right', fontsize=11)
        ax_bcc.grid(True, linestyle=':', alpha=0.6)

        fig_bcc.tight_layout()
        bcc_save_path = save_dir / f"plot_bcc_{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{group_name}_{period_name}.png"
        fig_bcc.savefig(bcc_save_path, dpi=300)

        # --- Formatting RMSE Plot ---
        ax_rmse.axhline(y=1.2, color='black', linestyle='--', linewidth=2)
        ax_rmse.axhline(y=np.sqrt(2), color='gray', linestyle='-.', linewidth=2)
        ax_rmse.set_xticks(np.arange(0, 41, 5))
        ax_rmse.set_xlim(0, 35)
        ax_rmse.set_ylim(0, 1.8)
        ax_rmse.set_xlabel('Forecast lead (days)')
        ax_rmse.set_ylabel('RMSE')
        # ax_rmse.set_title(f'RMSE Skill Comparison: {group_name.replace("_", " ")} ({period_name.upper()})')
        # ax_rmse.legend(loc='upper left', fontsize=11)
        ax_rmse.grid(True, linestyle=':', alpha=0.6)

        fig_rmse.tight_layout()
        rmse_save_path = save_dir / f"plot_rmse_{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{group_name}_{period_name}.png"
        fig_rmse.savefig(rmse_save_path, dpi=300)
        
        # Close figures to prevent memory leaks in the loop
        plt.close(fig_bcc)
        plt.close(fig_rmse)

print(f"\nAll comparative plots generated and saved successfully to:\n{save_dir}")
