import yaml
import os
import copy
import numpy as np
import sys
from pathlib import Path
scripts_path = Path(__file__).resolve().parents[2]
sys.path.append(str(scripts_path))
sys.path.append(str(scripts_path / 'src'))

from data_prepare.dataset import load_train_data, load_val_data
from models.cnnmlp import CNNMLP
from models.unet import UNet_A
from trainers.train import train_model_hpo 
from utils.logger import setup_logger
import optuna
import torch 
from optuna.storages import RDBStorage, RetryFailedTrialCallback
from optuna.trial import TrialState

# ======================================================================
# 1. Configuration & Environment Setup
# ======================================================================
CONFIG_PATH = "/scratch/users/linyao/ML4MJO/scripts/configs/MDbase.yaml"

with open(CONFIG_PATH, "r") as f:
    base_config = yaml.safe_load(f)

dataflg = os.environ.get("dataflg", "era5").lower()
input_dir = os.environ.get("expflg", "rescaled_m10resi_wnx9resi")
input_var_name = os.environ.get("input_var", "olr")
output_var_name = os.environ.get("output_var", "ROMI")
model_name = os.environ.get("model_name", "UNet_A")
study_id = int(os.environ.get("study_id", 1))
multi_lead = os.environ.get("multi_lead", "true").lower() == "true"

# --- Data Path Logic ---
if dataflg == "noaa":
    target_path = '/scratch/users/linyao/ML4MJO/data/targets/noaa/RMM_ERA5_daily_1980to2001.noaa.nc' if output_var_name == 'RMM' else '/scratch/users/linyao/ML4MJO/data/targets/noaa/ROMI_NOAA_1979to2022.nc'
    input_path = [f'/scratch/users/linyao/ML4MJO/data/{input_dir}/olr.fltano120.noaa.2x2.1979to2022based1979to2001.nc'] if input_var_name == "olr" else [f'/scratch/users/linyao/ML4MJO/data/{input_dir}/{input_var_name}.fltano120.1978to2025based1979to2001.nc']
else: # era5
    target_path = '/scratch/users/linyao/ML4MJO/data/targets/RMM_ERA5_daily_1979to2001.nc' if output_var_name == 'RMM' else '/scratch/users/linyao/ML4MJO/data/targets/ROMI_ERA5_daily_1979to2012.nc'
    input_path = [f'/scratch/users/linyao/ML4MJO/data/{input_dir}/{input_var_name}.fltano120.1978to2025based1979to2001.nc']

base_config["data"].update({
    "dataflg": dataflg,
    "input_vars": [input_var_name],
    "target_vars": [output_var_name],
    "input_path": input_path,
    "target_path": target_path,
    "lat_range": int(os.environ.get("lat_range", 20))
})

base_config["model"]["name"] = model_name
base_config["training"]["multi_lead"] = multi_lead
n_modes = 2 
if not multi_lead:
    lead = int(os.environ.get("lead", 0))
    base_config["data"]["lead"] = lead
    exp_name = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{input_dir}_lead{lead}"
else:
    lead = int(os.environ.get("lead", 35))
    if lead != 35:
        base_config["data"]["lead"] = lead
        exp_name = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{input_dir}_lead{lead}"
    else:
        exp_name = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{input_dir}"

os.makedirs('./yaml', exist_ok=True)
with open(f'./yaml/hpo_{exp_name}.yaml', 'w') as f:
    yaml.dump(base_config, f)

optuna_dir = "./optuna"
os.makedirs(optuna_dir, exist_ok=True)
storage_url = f"sqlite:///{os.path.join(optuna_dir, f'{exp_name}_s{study_id}.db')}"

# ======================================================================
# 2. Optuna Objective Function
# ======================================================================
def objective(trial):
    config = copy.deepcopy(base_config)

    # --- HPO: Normalization ---
    norm_type = trial.suggest_categorical("norm_type", ["batch", "group"])
    num_groups = 8 if norm_type == "group" else 0 
    
    # --- HPO: CNN Architecture ---
    if config["model"]["name"] == "CNN_MLP":
        n_filters = trial.suggest_int("num_filters_enc", 16, 64, step=16)
        config["model"]["cnn"]["channels_list"] = [n_filters, n_filters]
    else: # UNet_A
        config["model"]["cnn"]["num_filters_enc"] = trial.suggest_int("num_filters_enc", 16, 64, step=16)

    grid_res = float(config["data"]["grid_res"])
    lat_range = config["data"]["lat_range"]
    nlon = int(np.floor(360 / grid_res))
    nlat = int(np.floor(lat_range / grid_res) * 2 + 1)
    
    config["model"]["cnn"].update({
        "input_channel_num": len(config["data"]["input_path"]),
        "input_map_size": nlon * nlat,
        "norm_type": norm_type,
        "num_groups": num_groups,
        "dropout": 0.0 
    })
    
    k_size = trial.suggest_int("kernel_size", 3, 9, step=2)
    config["model"]["cnn"]["kernel_size"] = [k_size, k_size]

    # --- HPO: MLP Architecture ---
    lead = config["data"]["lead"]
    multi_lead = config["training"].get("multi_lead", True) 
    output_size = (lead + 1) * n_modes if multi_lead else n_modes
    
    h1 = trial.suggest_int("nhidden1", 128, 512, step=128)
    h2 = trial.suggest_int("nhidden2", 64, 256, step=64)
    h3 = trial.suggest_int("nhidden3", 16, 128, step=16)
    
    config["model"]["mlp"].update({
        "hidden_layers": [h1, h2, h3],
        "output_size": output_size,
        "dropout": trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    })

    # --- HPO: Training Params ---
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])
    
    if optimizer_name == "AdamW":
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        momentum = 0.9 # Default for AdamW
    else: # SGD
        lr = trial.suggest_float("learning_rate", 1e-3, 5e-1, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        momentum = trial.suggest_float("momentum", 0.8, 0.95, step=0.05)

    config["training"].update({
        "optimizer": optimizer_name,
        "learning_rate": lr,
        "weight_decay": wd,
        "momentum": momentum,
        "batch_size": trial.suggest_int("batch_size", 16, 64, step=16),
        "epochs": 20,
        "early_stopping_patience": 7 
    })

    print(f"\n{'='*20} TRIAL #{trial.number} {'='*20}")
    print(f"Params: {optimizer_name} | LR: {lr:.2e} | WD: {wd:.2e} | BS: {config['training']['batch_size']}")
    print(f"Arch: {h1}-{h2}-{h3} | Norm: {norm_type} | K: {k_size}")

    # Data Loading
    train_loader = load_train_data(config)
    val_loader = load_val_data(config)

    # Model Building
    if config["model"]["name"] == "CNN_MLP":
        model = CNNMLP(config["model"]["cnn"], config["model"]["mlp"])
    else:
        model = UNet_A(config["model"]["cnn"], config["model"]["mlp"])

    # Training
    try:
        val_loss, _ = train_model_hpo(model, train_loader, val_loader, config, trial)
        
        if np.isnan(val_loss):
            print(f"Trial {trial.number} resulted in NaN. Pruning.")
            raise optuna.TrialPruned()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            # Clear cache to free up the GPU for the next trial
            torch.cuda.empty_cache()
            print(f"Trial {trial.number} FAILED due to CUDA OOM. Pruning and moving to next trial.")
            # Tell Optuna this was a bad configuration so it avoids it in the future
            raise optuna.TrialPruned()
        else:
            # If it's a different error, we still want to see it
            raise e

    print(f"Final Val Loss: {val_loss:.6f}")

    return val_loss

# ======================================================================
# 3. Execution
# ======================================================================
if __name__ == "__main__":
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    # 1. Setup Robust Storage
    # Heartbeat tracks active trials. If a node is preempted, the heartbeat stops.
    # Optuna will notice the dead heartbeat and mark the trial as FAILED.
    # The callback will then automatically re-queue those exact parameters for a new trial.
    storage = RDBStorage(
        url=storage_url,
        heartbeat_interval=120,  # Record a heartbeat every 120s
        grace_period=240,       # Mark as FAILED if no heartbeat is seen for 240s
        failed_trial_callback=RetryFailedTrialCallback(max_retry=3)
    )

    study = optuna.create_study(
        study_name=f"{exp_name}_s{study_id}",
        storage=storage,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True
    )

    # 2. Manual Cleanup on Startup (For previously stuck trials)
    # This catches trials that were left RUNNING before you implemented heartbeats
    # or instances where the storage backend needs a manual push.
    stuck_trials = [t for t in study.trials if t.state == TrialState.RUNNING]
    for t in stuck_trials:
        print(f"Node restart detected. Re-queueing params from interrupted Trial {t.number}...")
        try:
            study.enqueue_trial(t.params)
        except ValueError:
            # Optuna might throw an error if the exact same params are already queued
            pass 

    n_total = 100
    n_completed = len([t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)])
    
    if n_completed < n_total:
        study.optimize(objective, n_trials=n_total - n_completed)
    
    print(f"Best Trial: {study.best_trial.number} | Value: {study.best_trial.value}")
