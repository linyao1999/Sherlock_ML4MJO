import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================================================================
# 1. Environment & Path Setup
# ======================================================================
dataflg = os.environ.get("dataflg", "era5").lower()
input_dir = os.environ.get("expflg", "fltano120")
input_var_name = os.environ.get("input_var", "olr")
output_var_name = os.environ.get("output_var", "RMM")
model_name = os.environ.get("model_name", "UNet_A")
multi_lead = os.environ.get("multi_lead", "false").lower() == "true"
single_lead = 15 if output_var_name == 'RMM' else 25

if not multi_lead:
    lead = int(os.environ.get("lead", single_lead))
    exp_name = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{input_dir}_lead{lead}"
else:
    exp_name = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{input_dir}"

save_dir = Path(f"/scratch/users/linyao/ML4MJO/scripts/outputs/contribution/{exp_name}")
fig_dir = Path("/scratch/users/linyao/ML4MJO/scripts/outputs/Figs/contribution")
fig_dir.mkdir(parents=True, exist_ok=True)

# Metrics directory (for loading BCC values to find the best experiment)
metrics_dir = Path(f"/scratch/users/linyao/ML4MJO/scripts/outputs/metrics/{exp_name}")
target_lead = single_lead  # lead=15 for RMM, lead=25 for ROMI

num_experiments = 16

# ======================================================================
# 2. Load Contribution Data for All Experiments
# ======================================================================
print(f"Loading ablation data for {exp_name} across {num_experiments} experiments...")

all_ranked_pct = []
all_unranked_pct = []
n_hid2 = None
n_hid6 = None

for exp_num in range(1, num_experiments + 1):
    file_path = save_dir / f"contribution_data_{exp_name}_exp{exp_num}.npz"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Expected {num_experiments} experiments but missing: {file_path}"
        )

    loaded_data = np.load(file_path)
    baseline_mse = loaded_data["baseline_mse"].item()

    if model_name == 'CNN_MLP':
        pct_change = (loaded_data["cnn_last"] / baseline_mse) * 100
    else:
        # UNet_A: Normalize both layers before concatenating
        pct_hid2 = (loaded_data["hid2"] / baseline_mse) * 100
        pct_hid6 = (loaded_data["hid6"] / baseline_mse) * 100
        pct_change = np.concatenate([pct_hid2, pct_hid6])
        if n_hid2 is None:
            n_hid2 = len(pct_hid2)
            n_hid6 = len(pct_hid6)

    # Sort descending to build the "Rank of importance" curve for this experiment
    ranked_pct = np.sort(pct_change)[::-1]
    all_ranked_pct.append(ranked_pct)
    all_unranked_pct.append(pct_change)

if not all_ranked_pct:
    print("No data found to plot. Exiting.")
    sys.exit()

# Shape: (N_experiments, N_channels)
all_ranked_pct = np.array(all_ranked_pct)
all_unranked_pct = np.array(all_unranked_pct)
print(f"Successfully loaded data for {all_ranked_pct.shape[0]} experiments.")

# ======================================================================
# 3. Load BCC Metrics to Find the Best Experiment
# ======================================================================
bcc_file = metrics_dir / f"metrics_{exp_name}_t1_test.npz"
if not bcc_file.exists():
    raise FileNotFoundError(f"BCC metrics file not found: {bcc_file}")

bcc_data = np.load(bcc_file)
ind_bcc = bcc_data["ind_bcc"]  # (16,) single_lead, (16, lead_max+1) multi_lead

if ind_bcc.ndim == 2:
    # Multi-lead metrics: select BCC at the target lead
    bcc_at_lead = ind_bcc[:, target_lead]
    lead_label = target_lead
else:
    bcc_at_lead = ind_bcc
    lead_label = lead if not multi_lead else target_lead

best_exp_idx = int(np.argmax(bcc_at_lead))  # 0-indexed
print(
    f"Best experiment (by BCC at lead={lead_label}): "
    f"exp{best_exp_idx + 1} with BCC={bcc_at_lead[best_exp_idx]:.4f}"
)

# ======================================================================
# 4. Plotting: Ranked Contribution
# ======================================================================
print("Plotting ensemble contribution analysis (ranked)...")
plt.rcParams['font.size'] = 24

plt.figure(figsize=(8.5, 6.5))
channels_rank = np.arange(all_ranked_pct.shape[1])

# Scatter each experiment's ranked contribution as low-alpha tab:blue dots
for i in range(num_experiments):
    plt.plot(
        channels_rank,
        all_ranked_pct[i],
        'o-',
        color='tab:blue',
        alpha=1,
        markersize=8,
    )

# Highlight the best-BCC experiment in orange
best_label = f'exp{best_exp_idx + 1}' if not multi_lead else None
plt.plot(
    channels_rank,
    all_ranked_pct[best_exp_idx],
    'o-',
    color='tab:orange',
    alpha=1,
    markersize=8,
    label=best_label,
)

plt.xlabel("Rank of importance")
plt.ylabel("Normalized output difference (%)")
plt.grid(True, linestyle='--', alpha=0.6)
if not multi_lead:
    plt.legend(fontsize=18, loc='best')
plt.tight_layout()

output_filename = fig_dir / f"contribution_analysis_{exp_name}_all16_ranked.png"
plt.savefig(output_filename, dpi=300)
plt.close()

print(f"Ranked plot saved successfully to: {output_filename}")

# ======================================================================
# 5. Plotting: Unranked Contribution (by channel index)
# ======================================================================
print("Plotting ensemble contribution analysis (unranked)...")

plt.figure(figsize=(8.5, 6.5))
channels_idx = np.arange(all_unranked_pct.shape[1])

# Scatter each experiment's raw contribution as low-alpha tab:blue dots
for i in range(num_experiments):
    plt.plot(
        channels_idx,
        all_unranked_pct[i],
        'o',
        color='tab:blue',
        alpha=1,
        markersize=8,
    )

# Highlight the best-BCC experiment in orange
best_label = f'exp{best_exp_idx + 1}' if not multi_lead else None
plt.plot(
    channels_idx,
    all_unranked_pct[best_exp_idx],
    'o',
    color='tab:orange',
    alpha=1,
    markersize=8,
    label=best_label,
)

# For UNet_A, mark the boundary between hid2 and hid6 channels
if model_name != 'CNN_MLP' and n_hid2 is not None:
    plt.axvline(x=n_hid2 - 0.5, color='k', linestyle='--', linewidth=1.5, alpha=0.7)

plt.xlabel("Channel number")
plt.ylabel("Normalized output difference (%)")
plt.grid(True, linestyle='--', alpha=0.6)
if not multi_lead:
    plt.legend(fontsize=18, loc='best')
plt.tight_layout()

output_filename_unranked = fig_dir / f"contribution_analysis_{exp_name}_all16_unranked.png"
plt.savefig(output_filename_unranked, dpi=300)
plt.close()

print(f"Unranked plot saved successfully to: {output_filename_unranked}")

# ======================================================================
# 6. Plotting: Multi-lead Comparison (best BCC experiment per lead)
# ======================================================================
print("Plotting multi-lead contribution comparison...")

base_exp_name = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{input_dir}"
compare_leads = [0, 5, 10, 15, 20, 25]
lead_colors = plt.cm.viridis(np.linspace(0, 0.9, len(compare_leads)))

fig_multi_ranked, ax_multi_ranked = plt.subplots(figsize=(8.5, 6.5))
fig_multi_unranked, ax_multi_unranked = plt.subplots(figsize=(8.5, 6.5))

for lead_idx, lead_val in enumerate(compare_leads):
    lead_exp_name = f"{base_exp_name}_lead{lead_val}"
    lead_contrib_dir = Path(
        f"/scratch/users/linyao/ML4MJO/scripts/outputs/contribution/{lead_exp_name}"
    )
    lead_metrics_dir = Path(
        f"/scratch/users/linyao/ML4MJO/scripts/outputs/metrics/{lead_exp_name}"
    )

    # Load contribution data for all 16 experiments at this lead
    lead_all_ranked = []
    lead_all_unranked = []
    skip_lead = False
    for exp_num in range(1, num_experiments + 1):
        cf = lead_contrib_dir / f"contribution_data_{lead_exp_name}_exp{exp_num}.npz"
        if not cf.exists():
            print(
                f"  [Warn] Missing contribution file for lead={lead_val}: {cf}. "
                f"Skipping this lead."
            )
            skip_lead = True
            break
        d = np.load(cf)
        b = d["baseline_mse"].item()
        if model_name == 'CNN_MLP':
            pc = (d["cnn_last"] / b) * 100
        else:
            pc = np.concatenate(
                [(d["hid2"] / b) * 100, (d["hid6"] / b) * 100]
            )
        lead_all_ranked.append(np.sort(pc)[::-1])
        lead_all_unranked.append(pc)

    if skip_lead:
        continue

    lead_all_ranked = np.array(lead_all_ranked)
    lead_all_unranked = np.array(lead_all_unranked)

    # Load BCC for this lead to find the best experiment
    bcc_f = lead_metrics_dir / f"metrics_{lead_exp_name}_t1_test.npz"
    if not bcc_f.exists():
        print(
            f"  [Warn] Missing BCC file for lead={lead_val}: {bcc_f}. "
            f"Skipping this lead."
        )
        continue

    bd = np.load(bcc_f)
    ib = bd["ind_bcc"]
    if ib.ndim == 2:
        ib = ib[:, lead_val]
    best_idx_lead = int(np.argmax(ib))

    ranked_best = lead_all_ranked[best_idx_lead]
    unranked_best = lead_all_unranked[best_idx_lead]

    cr = np.arange(len(ranked_best))
    ci = np.arange(len(unranked_best))

    ax_multi_ranked.plot(
        cr,
        ranked_best,
        'o-',
        color=lead_colors[lead_idx],
        alpha=1,
        markersize=8,
        label=f'lead={lead_val} (exp{best_idx_lead + 1})',
    )
    ax_multi_unranked.plot(
        ci,
        unranked_best,
        'o',
        color=lead_colors[lead_idx],
        alpha=1,
        markersize=8,
        label=f'lead={lead_val} (exp{best_idx_lead + 1})',
    )

ax_multi_ranked.set_xlabel("Rank of importance")
ax_multi_ranked.set_ylabel("Normalized output difference (%)")
ax_multi_ranked.grid(True, linestyle='--', alpha=0.6)
ax_multi_ranked.legend(fontsize=12, loc='best')
fig_multi_ranked.tight_layout()

ax_multi_unranked.set_xlabel("Channel number")
ax_multi_unranked.set_ylabel("Normalized output difference (%)")
if model_name != 'CNN_MLP' and n_hid2 is not None:
    ax_multi_unranked.axvline(
        x=n_hid2 - 0.5, color='k', linestyle='--', linewidth=1.5, alpha=0.7
    )
ax_multi_unranked.grid(True, linestyle='--', alpha=0.6)
ax_multi_unranked.legend(fontsize=12, loc='best')
fig_multi_unranked.tight_layout()

fn_multi_ranked = fig_dir / f"contribution_analysis_{base_exp_name}_multilead_ranked.png"
fn_multi_unranked = fig_dir / f"contribution_analysis_{base_exp_name}_multilead_unranked.png"
fig_multi_ranked.savefig(fn_multi_ranked, dpi=300)
fig_multi_unranked.savefig(fn_multi_unranked, dpi=300)
plt.close(fig_multi_ranked)
plt.close(fig_multi_unranked)

print(f"Multi-lead ranked plot saved to: {fn_multi_ranked}")
print(f"Multi-lead unranked plot saved to: {fn_multi_unranked}")
