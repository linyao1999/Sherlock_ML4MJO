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
reload_model = os.environ.get("reload_model", "fltano120")
best_expflg = os.environ.get("best_expflg", "BEST")

# Multi-lead toggles
multi_lead = os.environ.get("multi_lead", "true").lower() == "true"
max_lead = int(os.environ.get("max_lead", 35))  # Used for concatenating single leads

# Assuming we are plotting Trial 1 based on your previous TOP_TRIALS = [1,]
trial_tag = "t1"
PERIODS = ["test"]

# Define the two groups of experiments
# Each entry is (solid_expflg, reload_expflg or None)
# Solid lines: original experiments (trained and evaluated on the same data)
# Dashed lines: reload experiments (model from reload_model, evaluated on that data)
groups = {
    "Rescaled_Reload": [
        ('fltano120', None),
        ('rescaled_m10all_wnx1off', f'model_{reload_model}_input_rescaled_m10all_wnx1off'),
        ('rescaled_m10all_wnx9all', f'model_{reload_model}_input_rescaled_m10all_wnx9all'),
        ('rescaled_m10resi_wnx9resi', f'model_{reload_model}_input_rescaled_m10resi_wnx9resi'),
    ],
    "Unscaled_Reload": [
        ('fltano120', None),
        ('unscaled_m10all_wnx1off', f'model_{reload_model}_input_unscaled_m10all_wnx1off'),
        ('unscaled_m10all_wnx9all', f'model_{reload_model}_input_unscaled_m10all_wnx9all'),
        ('unscaled_m10resi_wnx9resi', f'model_{reload_model}_input_unscaled_m10resi_wnx9resi'),
    ]
}

# The colors you requested ('gold' is used for 'yellow' to ensure visibility)
COLORS = ['green', 'gold', 'orange', 'red']

# Base directory where your previous script saved the metrics
metrics_base_dir = Path("/scratch/users/linyao/ML4MJO/scripts/outputs/metrics")

# Directory to save these new combined comparison plots
save_dir = Path("/scratch/users/linyao/ML4MJO/scripts/outputs/Figs") / "Fig2"
save_dir.mkdir(parents=True, exist_ok=True)


# ======================================================================
# 2. Helper: Load metrics for a given experiment flag
# ======================================================================
def load_metrics(expflg, period_name):
    """Load BCC/RMSE metrics for a given experiment flag.
    Returns (leads, ens_bcc, ind_bcc, ens_rmse, ind_rmse) or None."""
    exp_name = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{expflg}"

    if multi_lead:
        # --- Multi-lead logic (loads a single .npz file) ---
        data_path = metrics_base_dir / exp_name / f"metrics_{exp_name}_{trial_tag}_{period_name}.npz"
        if not data_path.exists():
            print(f"     [Warning] Missing data for {exp_name}. Skipping line.")
            return None
        data = np.load(data_path)
        lead_max = int(data['lead'])
        leads = np.arange(0, lead_max + 1)
        return leads, data['ens_bcc'], data['ind_bcc'], data['ens_rmse'], data['ind_rmse']
    else:
        # --- Single-lead logic (concatenates multiple .npz files) ---
        ens_bcc_list = []
        ind_bcc_list = []
        ens_rmse_list = []
        ind_rmse_list = []
        leads_avail = []

        for l in range(max_lead + 1):
            exp_name_lead = f"{exp_name}_lead{l}"
            data_path = metrics_base_dir / exp_name_lead / f"metrics_{exp_name_lead}_{trial_tag}_{period_name}.npz"

            if data_path.exists():
                data = np.load(data_path)
                ens_bcc_list.append(data['ens_bcc'])
                ens_rmse_list.append(data['ens_rmse'])
                ind_bcc_list.append(data['ind_bcc'])
                ind_rmse_list.append(data['ind_rmse'])
                leads_avail.append(l)

        if not leads_avail:
            print(f"     [Warning] No single-lead data found for {expflg}. Skipping line.")
            return None

        leads = np.array(leads_avail)
        ens_bcc = np.array(ens_bcc_list)
        ens_rmse = np.array(ens_rmse_list)

        # Transpose the individual arrays so shape is (N_Ensembles, Leads)
        ind_bcc = np.array(ind_bcc_list).T
        ind_rmse = np.array(ind_rmse_list).T
        return leads, ens_bcc, ind_bcc, ens_rmse, ind_rmse


def load_best_metrics(period_name):
    """Load BCC/RMSE metrics for the BEST experiment.
    BEST naming: {dataflg}_BEST_{model_name}_{output_var_name}
    Returns (leads, ens_bcc, ind_bcc, ens_rmse, ind_rmse) or None."""
    best_exp_name = f"{dataflg}_{best_expflg}_{model_name}_{output_var_name}"

    if multi_lead:
        data_path = metrics_base_dir / best_exp_name / f"metrics_{best_exp_name}_{trial_tag}_{period_name}.npz"
        if not data_path.exists():
            print(f"     [Warning] Missing BEST data: {data_path}. Skipping line.")
            return None
        data = np.load(data_path)
        lead_max = int(data['lead'])
        leads = np.arange(0, lead_max + 1)
        return leads, data['ens_bcc'], data['ind_bcc'], data['ens_rmse'], data['ind_rmse']
    else:
        ens_bcc_list = []
        ind_bcc_list = []
        ens_rmse_list = []
        ind_rmse_list = []
        leads_avail = []

        for l in range(max_lead + 1):
            best_exp_lead = f"{best_exp_name}_lead{l}"
            data_path = metrics_base_dir / best_exp_lead / f"metrics_{best_exp_lead}_{trial_tag}_{period_name}.npz"
            if data_path.exists():
                data = np.load(data_path)
                ens_bcc_list.append(data['ens_bcc'])
                ens_rmse_list.append(data['ens_rmse'])
                ind_bcc_list.append(data['ind_bcc'])
                ind_rmse_list.append(data['ind_rmse'])
                leads_avail.append(l)

        if not leads_avail:
            print(f"     [Warning] No single-lead BEST data found. Skipping line.")
            return None

        leads = np.array(leads_avail)
        ens_bcc = np.array(ens_bcc_list)
        ens_rmse = np.array(ens_rmse_list)
        ind_bcc = np.array(ind_bcc_list).T
        ind_rmse = np.array(ind_rmse_list).T
        return leads, ens_bcc, ind_bcc, ens_rmse, ind_rmse


# ======================================================================
# 3. Plotting Loop
# ======================================================================
print("\n" + "="*50)
print("=== GENERATING MULTI-EXPERIMENT PLOTS WITH RELOAD ===")
print("="*50)

# Slightly smaller font to accommodate the longer legend labels
plt.rcParams['font.size'] = 22

for group_name, experiments in groups.items():
    print(f"\nProcessing {group_name}...")

    for period_name in PERIODS:
        print(f"  -> Generating plots for period: {period_name.upper()}")

        fig_bcc, ax_bcc = plt.subplots(1, 1, figsize=(8.5, 6.5))
        fig_rmse, ax_rmse = plt.subplots(1, 1, figsize=(8.5, 6.5))

        # --- Plot BEST baseline (solid blue line) ---
        best_result = load_best_metrics(period_name)
        if best_result is not None:
            leads_b, ens_bcc_b, ind_bcc_b, ens_rmse_b, ind_rmse_b = best_result
            bcc_min_b, bcc_max_b = np.min(ind_bcc_b, axis=0), np.max(ind_bcc_b, axis=0)
            rmse_min_b, rmse_max_b = np.min(ind_rmse_b, axis=0), np.max(ind_rmse_b, axis=0)

            ax_bcc.fill_between(leads_b, bcc_min_b, bcc_max_b, color='tab:blue', alpha=0.15)
            ax_bcc.plot(leads_b, ens_bcc_b, '-o', color='tab:blue', linewidth=2, label='BEST')

            ax_rmse.fill_between(leads_b, rmse_min_b, rmse_max_b, color='tab:blue', alpha=0.15)
            ax_rmse.plot(leads_b, ens_rmse_b, '-o', color='tab:blue', linewidth=2, label='BEST')

        for i, (solid_expflg, reload_expflg) in enumerate(experiments):

            # --- Plot solid line (original experiment) ---
            result = load_metrics(solid_expflg, period_name)
            if result is not None:
                leads, ens_bcc, ind_bcc, ens_rmse, ind_rmse = result

                # Calculate ensemble spread / uncertainty
                bcc_min, bcc_max = np.min(ind_bcc, axis=0), np.max(ind_bcc, axis=0)
                rmse_min, rmse_max = np.min(ind_rmse, axis=0), np.max(ind_rmse, axis=0)

                # Plot BCC (Shading + Ensemble Mean)
                ax_bcc.fill_between(leads, bcc_min, bcc_max, color=COLORS[i], alpha=0.15)
                ax_bcc.plot(leads, ens_bcc, '-o', color=COLORS[i], linewidth=2, label=solid_expflg)

                # Plot RMSE (Shading + Ensemble Mean)
                ax_rmse.fill_between(leads, rmse_min, rmse_max, color=COLORS[i], alpha=0.15)
                ax_rmse.plot(leads, ens_rmse, '-o', color=COLORS[i], linewidth=2, label=solid_expflg)

            # --- Plot dashed line (reload experiment) ---
            if reload_expflg is not None:
                result = load_metrics(reload_expflg, period_name)
                if result is not None:
                    leads, ens_bcc, ind_bcc, ens_rmse, ind_rmse = result

                    bcc_min, bcc_max = np.min(ind_bcc, axis=0), np.max(ind_bcc, axis=0)
                    rmse_min, rmse_max = np.min(ind_rmse, axis=0), np.max(ind_rmse, axis=0)

                    ax_bcc.fill_between(leads, bcc_min, bcc_max, color=COLORS[i], alpha=0.08)
                    ax_bcc.plot(leads, ens_bcc, '--', color=COLORS[i], linewidth=2)

                    ax_rmse.fill_between(leads, rmse_min, rmse_max, color=COLORS[i], alpha=0.08)
                    ax_rmse.plot(leads, ens_rmse, '--', color=COLORS[i], linewidth=2)

        # --- Formatting BCC Plot ---
        ax_bcc.axhline(y=0.5, color='black', linestyle='--', linewidth=2)
        ax_bcc.set_xticks(np.arange(0, 41, 5))
        ax_bcc.set_xlim(0, 35)
        ax_bcc.set_ylim(0.1, 1.0)
        ax_bcc.set_yticks(np.arange(0.1, 1.1, 0.2))
        ax_bcc.set_xlabel('Forecast lead (days)')
        ax_bcc.set_ylabel('BCC')
        ax_bcc.grid(True, linestyle=':', alpha=0.6)

        fig_bcc.tight_layout()
        if multi_lead:
            bcc_save_path = save_dir / f"plot_bcc_reload_{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{group_name}_{period_name}_multi.pdf"
        else:
            bcc_save_path = save_dir / f"plot_bcc_reload_{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{group_name}_{period_name}_single.pdf"

        fig_bcc.savefig(bcc_save_path, dpi=300)

        # --- Formatting RMSE Plot ---
        # ax_rmse.axhline(y=1.2, color='black', linestyle='--', linewidth=2)
        # ax_rmse.axhline(y=np.sqrt(2), color='gray', linestyle='-.', linewidth=2)
        ax_rmse.set_xticks(np.arange(0, 41, 5))
        ax_rmse.set_xlim(0, 35)
        ax_rmse.set_ylim(0, 1.8)
        ax_rmse.set_xlabel('Forecast lead (days)')
        ax_rmse.set_ylabel('RMSE')
        ax_rmse.grid(True, linestyle=':', alpha=0.6)

        fig_rmse.tight_layout()
        if multi_lead:
            rmse_save_path = save_dir / f"plot_rmse_reload_{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{group_name}_{period_name}_multi.pdf"
        else:
            rmse_save_path = save_dir / f"plot_rmse_reload_{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{group_name}_{period_name}_single.pdf"

        fig_rmse.savefig(rmse_save_path, dpi=300)

        # Close figures to prevent memory leaks in the loop
        plt.close(fig_bcc)
        plt.close(fig_rmse)

print(f"\nAll comparative plots generated and saved successfully to:\n{save_dir}")
