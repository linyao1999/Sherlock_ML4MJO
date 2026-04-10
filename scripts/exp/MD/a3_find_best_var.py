import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================================
# 1. Configuration & Setup
# ======================================================================
# Variables to evaluate
input_vars = [
    'olr', 'tcwv', 'prep', 'sst', 'u200', 'u500', 'u850', 
    'q200', 'q500', 'q850', 'T200', 'T500', 'T850', 'v200', 
    'v500', 'v850', 'Z200', 'Z500', 'Z850'
]
output_vars = ['RMM', 'ROMI']
periods = ['test', 'val']
trial_rank = 1

# Base environment parameters 
dataflg = os.environ.get("dataflg", "era5").lower()
input_dir = os.environ.get("expflg", "fltano120")
model_name = os.environ.get("model_name", "UNet_A")
# Read the multi_lead flag just like in your main script
multi_lead = os.environ.get("multi_lead", "false").lower() == "true"

# Setup paths
base_metrics_dir = Path("/scratch/users/linyao/ML4MJO/scripts/outputs/metrics")
save_dir = Path("/scratch/users/linyao/ML4MJO/scripts/outputs/Figs/BestVar")
save_dir.mkdir(parents=True, exist_ok=True)

# Plot styling
plt.rcParams['font.size'] = 22

# ======================================================================
# 2. Data Extraction & Plotting Loop
# ======================================================================
print("\n" + "="*50)
print(f"=== GENERATING COMPARISON PLOTS (multi_lead: {multi_lead}) ===")
print("="*50)

for output_var in output_vars:
    # Set the target lead based on output variable
    target_lead = 15 if output_var == 'RMM' else 25
    
    for period in periods:
        print(f"\nProcessing {output_var} | Lead {target_lead} | Period: {period.upper()}")
        
        valid_vars = []
        bcc_ind_all, bcc_ens_all = [], []
        rmse_ind_all, rmse_ens_all = [], []
        
        # 1. Load Data
        for var in input_vars:
            
            # Adjust the expected experiment name based on the multi_lead flag
            if multi_lead:
                exp_name = f"{dataflg}_{var}_{model_name}_{output_var}_{input_dir}"
            else:
                exp_name = f"{dataflg}_{var}_{model_name}_{output_var}_{input_dir}_lead{target_lead}"
                
            file_path = base_metrics_dir / exp_name / f"metrics_{exp_name}_t{trial_rank}_{period}.npz"
            
            if os.path.exists(file_path):
                try:
                    data = np.load(file_path)
                    
                    # 2. Extract metrics based on array shapes
                    if multi_lead:
                        # Data contains all leads; slice to get the target lead
                        bcc_ind_all.append(data['ind_bcc'][:, target_lead])
                        bcc_ens_all.append(data['ens_bcc'][target_lead])
                        rmse_ind_all.append(data['ind_rmse'][:, target_lead])
                        rmse_ens_all.append(data['ens_rmse'][target_lead])
                    else:
                        # Data only contains the specific lead evaluated in that job
                        bcc_ind_all.append(data['ind_bcc'])
                        bcc_ens_all.append(data['ens_bcc'])
                        rmse_ind_all.append(data['ind_rmse'])
                        rmse_ens_all.append(data['ens_rmse'])
                    
                    valid_vars.append(var.upper())
                except Exception as e:
                    print(f"  [Warning] Error reading {file_path}: {e}")
            else:
                print(f"  [Missing] {file_path}")
        
        if not valid_vars:
            print(f"  -> No data found for {output_var} {period}. Skipping plot.")
            continue
            
        x_positions = np.arange(len(valid_vars))
        
        # ==========================================
        # Plot 1: BCC
        # ==========================================
        fig_bcc, ax_bcc = plt.subplots(figsize=(14, 5))
        
        for i, x in enumerate(x_positions):
            # Plot individuals in gray
            ax_bcc.scatter([x] * len(bcc_ind_all[i]), bcc_ind_all[i], 
                           color='gray', alpha=0.7, s=150, zorder=2)
            # Plot ensemble mean in blue
            ax_bcc.scatter(x, bcc_ens_all[i], 
                           color='tab:blue', s=150, edgecolor='black', zorder=3)

        # Formatting BCC plot
        ax_bcc.set_xticks(x_positions)
        ax_bcc.set_xticklabels(valid_vars, rotation=60, ha='center')
        # ax_bcc.axhline(0.5, color='black', linestyle='--', zorder=1) # Reference line
        ax_bcc.axhline(np.max(bcc_ens_all), color='tab:blue', linestyle='--', zorder=1) # Reference line
        # ax_bcc.set_ylabel('BCC')
        # ax_bcc.set_title(f'{output_var} BCC Comparison at Lead {target_lead} ({period.upper()})')
        
        plt.tight_layout()
        if multi_lead:
            bcc_save_path = save_dir / f"scatter_bcc_{dataflg}_{output_var}_multi_{period}.png"
        else:
            bcc_save_path = save_dir / f"scatter_bcc_{dataflg}_{output_var}_single_{period}.png"

        fig_bcc.savefig(bcc_save_path, dpi=300)
        plt.close(fig_bcc)
        print(f"  -> Saved BCC plot: {bcc_save_path.name}")
        
        # ==========================================
        # Plot 2: RMSE
        # ==========================================
        fig_rmse, ax_rmse = plt.subplots(figsize=(14, 5))
        
        for i, x in enumerate(x_positions):
            # Plot individuals in gray
            ax_rmse.scatter([x] * len(rmse_ind_all[i]), rmse_ind_all[i], 
                            color='gray', alpha=0.7, s=150, zorder=2)
            # Plot ensemble mean in blue
            ax_rmse.scatter(x, rmse_ens_all[i], 
                            color='tab:blue', s=150, edgecolor='black', zorder=3)

        # Formatting RMSE plot
        ax_rmse.set_xticks(x_positions)
        ax_rmse.set_xticklabels(valid_vars, rotation=60, ha='center')
        
        ax_rmse.axhline(1.2, color='black', linestyle='--', zorder=1)
        ax_rmse.axhline(np.sqrt(2), color='gray', linestyle='-.', zorder=1)
        
        # ax_rmse.set_ylabel('RMSE')
        # ax_rmse.set_title(f'{output_var} RMSE Comparison at Lead {target_lead} ({period.upper()})')
        
        plt.tight_layout()
        if multi_lead:
            rmse_save_path = save_dir / f"scatter_rmse_{dataflg}_{output_var}_multi_{period}.png"
        else:
            rmse_save_path = save_dir / f"scatter_rmse_{dataflg}_{output_var}_single_{period}.png"
        fig_rmse.savefig(rmse_save_path, dpi=300)
        plt.close(fig_rmse)
        print(f"  -> Saved RMSE plot: {rmse_save_path.name}")

print("\nAll comparison plots generated successfully in: ", save_dir)