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
output_var_name = os.environ.get("output_var", "RMM")
model_name = os.environ.get("model_name", "UNet_A")
rule = os.environ.get("rule", 'Iamp>1.0')
# rule = os.environ.get("rule", 'All')
# rule = os.environ.get("rule", 'phase<3')

# Multi-lead toggles
multi_lead = os.environ.get("multi_lead", "true").lower() == "true"
max_lead = int(os.environ.get("max_lead", 35)) # Used for concatenating single leads

# Assuming we are plotting Trial 1 based on your previous TOP_TRIALS = [1,]
trial_tag = "t1" 
PERIODS = ["test"]

# Define the two groups of experiments
groups = {
    # "Rescaled_Group": [
    #     'fltano120', 
    #     'rescaled_m10all_wnx1off', 
    #     'rescaled_m10all_wnx9all', 
    #     'rescaled_m10resi_wnx9resi'
    # ],
    "Unscaled_Group": [
        'fltano120', 
        'unscaled_m10all_wnx1off_c20', 
        'unscaled_m10all_wnx9all_c20', 
        'unscaled_m10resi_wnx9resi_c20'
    ]
}

# The colors you requested ('gold' is used for 'yellow' to ensure visibility)
COLORS = ['green', 'gold', 'orange', 'red']

# Base directory where your previous script saved the metrics
metrics_base_dir = Path("/scratch/users/linyao/ML4MJO/scripts/outputs/metrics")

# Directory to save these new combined comparison plots
save_dir = Path("/scratch/users/linyao/ML4MJO/scripts/outputs/Figs")
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
            
            if multi_lead:
                # --- Multi-lead logic (loads a single .npz file) ---
                exp_name = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{expflg}"
                if rule != 'Iamp>1.0':
                    data_path = metrics_base_dir / exp_name / f"metrics_{exp_name}_{trial_tag}_{period_name}_{rule}.npz"
                else:
                    if expflg == 'fltano120':
                        data_path = metrics_base_dir / exp_name / f"metrics_{exp_name}_{trial_tag}_{period_name}.npz"
                    else:
                        data_path = metrics_base_dir / exp_name / f"metrics_sens_{exp_name}_{trial_tag}_{period_name}.npz"

                if not data_path.exists():
                    print(f"     [Warning] Missing data for {exp_name}. Skipping line.")
                    continue
                    
                data = np.load(data_path)
                lead_max = int(data['lead'])
                leads = np.arange(0, lead_max + 1)
                
                ens_bcc, ind_bcc = data['ens_bcc'], data['ind_bcc']
                ens_rmse, ind_rmse = data['ens_rmse'], data['ind_rmse']
            
            else:
                # --- Single-lead logic (concatenates multiple .npz files) ---
                ens_bcc_list = []
                ind_bcc_list = []
                ens_rmse_list = []
                ind_rmse_list = []
                leads_avail = []

                for l in range(max_lead + 1):
                    exp_name_lead = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{expflg}_lead{l}"
                    if rule != 'Iamp>1.0':
                        data_path = metrics_base_dir / exp_name_lead / f"metrics_{exp_name_lead}_{trial_tag}_{period_name}_{rule}.npz"
                    else:
                        data_path = metrics_base_dir / exp_name_lead / f"metrics_{exp_name_lead}_{trial_tag}_{period_name}.npz"

                    if data_path.exists():
                        data = np.load(data_path)
                        # Append scalars
                        ens_bcc_list.append(data['ens_bcc'])
                        ens_rmse_list.append(data['ens_rmse'])
                        # Append individual member arrays
                        ind_bcc_list.append(data['ind_bcc'])
                        ind_rmse_list.append(data['ind_rmse'])
                        leads_avail.append(l)
                
                if not leads_avail:
                    print(f"     [Warning] No single-lead data found for {expflg}. Skipping line.")
                    continue
                
                leads = np.array(leads_avail)
                ens_bcc = np.array(ens_bcc_list)
                ens_rmse = np.array(ens_rmse_list)
                
                # Transpose the individual arrays so shape is (N_Ensembles, Leads)
                # This guarantees np.min(ind_bcc, axis=0) gives the min value per lead
                ind_bcc = np.array(ind_bcc_list).T
                ind_rmse = np.array(ind_rmse_list).T

            # Calculate ensemble spread / uncertainty
            bcc_min, bcc_max = np.min(ind_bcc, axis=0), np.max(ind_bcc, axis=0)
            rmse_min, rmse_max = np.min(ind_rmse, axis=0), np.max(ind_rmse, axis=0)
            
            # Plot BCC (Shading + Ensemble Mean)
            ax_bcc.fill_between(leads, bcc_min, bcc_max, color=COLORS[i], alpha=0.15)
            ax_bcc.plot(leads, ens_bcc, '-o', color=COLORS[i], linewidth=2, label=expflg)
            
            # Plot RMSE (Shading + Ensemble Mean)
            ax_rmse.fill_between(leads, rmse_min, rmse_max, color=COLORS[i], alpha=0.15)
            ax_rmse.plot(leads, ens_rmse, '-o', color=COLORS[i], linewidth=2, label=expflg)

        # --- Formatting BCC Plot ---
        ax_bcc.axhline(y=0.5, color='black', linestyle='--', linewidth=2)
        ax_bcc.set_xticks(np.arange(0, 41, 5))
        ax_bcc.set_xlim(0, 35)
        ax_bcc.set_ylim(0.1, 1.0)
        ax_bcc.set_yticks(np.arange(0.1, 1.1, 0.2))
        ax_bcc.set_xlabel('Forecast lead (days)')
        ax_bcc.set_ylabel('BCC')
        # Uncomment below if you want titles/legends back:
        # ax_bcc.set_title(f'BCC Skill Comparison: {group_name.replace("_", " ")} ({period_name.upper()})')
        # ax_bcc.legend(loc='upper right', fontsize=11)
        ax_bcc.grid(True, linestyle=':', alpha=0.6)

        fig_bcc.tight_layout()
        if multi_lead:
            if rule != 'Iamp>1.0':
                bcc_save_path = save_dir / f"plot_sens_c20_bcc_{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{group_name}_{period_name}_{rule}_multi.png"
            else:
                bcc_save_path = save_dir / f"plot_sens_c20_bcc_{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{group_name}_{period_name}_multi.png"
        else:
            if rule != 'Iamp>1.0':
                bcc_save_path = save_dir / f"plot_sens_c20_bcc_{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{group_name}_{period_name}_{rule}_single.png"
            else:
                bcc_save_path = save_dir / f"plot_sens_c20_bcc_{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{group_name}_{period_name}_single.png"

        fig_bcc.savefig(bcc_save_path, dpi=300)

        # --- Formatting RMSE Plot ---
        ax_rmse.axhline(y=1.2, color='black', linestyle='--', linewidth=2)
        ax_rmse.axhline(y=np.sqrt(2), color='gray', linestyle='-.', linewidth=2)
        ax_rmse.set_xticks(np.arange(0, 41, 5))
        ax_rmse.set_xlim(0, 35)
        ax_rmse.set_ylim(0, 1.8)
        ax_rmse.set_xlabel('Forecast lead (days)')
        ax_rmse.set_ylabel('RMSE')
        # Uncomment below if you want titles/legends back:
        # ax_rmse.set_title(f'RMSE Skill Comparison: {group_name.replace("_", " ")} ({period_name.upper()})')
        # ax_rmse.legend(loc='upper left', fontsize=11)
        ax_rmse.grid(True, linestyle=':', alpha=0.6)
        
        fig_rmse.tight_layout()
        if multi_lead:
            if rule != 'Iamp>1.0':
                rmse_save_path = save_dir / f"plot_sens_c20_rmse_{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{group_name}_{period_name}_{rule}_multi.png"
            else:
                rmse_save_path = save_dir / f"plot_sens_c20_rmse_{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{group_name}_{period_name}_multi.png"
        else:
            if rule != 'Iamp>1.0':
                rmse_save_path = save_dir / f"plot_sens_c20_rmse_{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{group_name}_{period_name}_{rule}_single.png"
            else:
                rmse_save_path = save_dir / f"plot_sens_c20_rmse_{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{group_name}_{period_name}_single.png"

        fig_rmse.savefig(rmse_save_path, dpi=300)
        
        # Close figures to prevent memory leaks in the loop
        plt.close(fig_bcc)
        plt.close(fig_rmse)

print(f"\nAll comparative plots generated and saved successfully to:\n{save_dir}")
