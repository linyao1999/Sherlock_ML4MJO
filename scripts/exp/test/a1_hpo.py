import yaml
import os
import copy
import numpy as np
import sys
from pathlib import Path
from math import ceil

# Add src/ to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2] / 'src'))

from data_prepare.dataset import load_train_data, load_val_data
from models.cnnmlp import CNNMLP
from models.unet import UNet_A
from trainers.train import train_model_hpo  # Using your new refactored version
from utils.logger import setup_logger

import optuna

logger = setup_logger()

# ======================================================================
# 1. Configuration & Environment Setup
# ======================================================================
CONFIG_PATH = "/scratch/users/linyao/ML4MJO/scripts/rescaled_input_2026/config/MDbase.yaml"

with open(CONFIG_PATH, "r") as f:
    base_config = yaml.safe_load(f)

# Modify base config from Environment Variables
input_dir = os.environ.get("expflg", "fltano120")
input_var_name = os.environ.get("input_var", "olr")
output_var_name = os.environ.get("output_var", "ROMI")
model_name = os.environ.get("model_name", "CNN_MLP")
batch_norm_enabled = os.environ.get("batch_norm", "false").lower() == "true"
study_id = int(os.environ.get("study_id", 1))

base_config["data"].update({
    "dataflg": input_dir,
    "input_vars": [input_var_name],
    "target_vars": [output_var_name],
    "input_path": [f'/scratch/users/linyao/ML4MJO/data/{input_dir}/{input_var_name}.fltano120.1978to2025based1979to2001.nc'],
    "lat_range": 20
})

if output_var_name == 'RMM':
    base_config["data"]["target_path"] = '/scratch/users/linyao/ML4MJO/data/targets/noaa/RMM_ERA5_daily_1980to2001.noaa.nc'
elif output_var_name == 'ROMI':
    base_config["data"]["target_path"] = '/scratch/users/linyao/ML4MJO/data/targets/noaa/ROMI_NOAA_1979to2022.nc'

base_config["model"]["name"] = model_name
base_config["model"]["cnn"]["batch_norm"] = batch_norm_enabled
base_config["training"]["early_stopping_patience"] = 10

# Naming Logic
bn_suffix = "_bn" if batch_norm_enabled else ""
exp_name = f"{input_var_name}_rescaled_{model_name}_{output_var_name}_{input_dir}{bn_suffix}"

os.makedirs('./yaml', exist_ok=True)
with open(f'./yaml/hpo_{exp_name}.yaml', 'w') as f:
    yaml.dump(base_config, f)

# Optuna Database Setup
study_name = f"{exp_name}_s{study_id}"
optuna_dir = "./optuna"
os.makedirs(optuna_dir, exist_ok=True)
storage = f"sqlite:///{os.path.join(optuna_dir, f'{study_name}.db')}"

print(f"🚀 Prepare done! Study: {study_name}")

# ======================================================================
# 2. Optuna Objective Function
# ======================================================================
def objective(trial):
    config = copy.deepcopy(base_config)

    # --- HPO: CNN Architecture ---
    if config["model"]["name"] == "CNN_MLP":
        n_filters = trial.suggest_int("num_filters_enc", 16, 64, step=16)
        config["model"]["cnn"]["channels_list"] = [n_filters, n_filters]
    elif config["model"]["name"] == "UNet_A":
        config["model"]["cnn"]["num_filters_enc"] = trial.suggest_int("num_filters_enc", 16, 64, step=16)

    # Grid Resolution / Spatial Dimensions
    grid_res = float(config["data"]["grid_res"])
    lat_range = config["data"]["lat_range"]
    nlon = int(np.floor(360 / grid_res))
    nlat = int(np.floor(lat_range / grid_res) * 2 + 1)
    
    config["model"]["cnn"].update({"nlon": nlon, "nlat": nlat, "input_map_size": nlon * nlat})
    
    k_size = trial.suggest_int("kernel_size", 3, 7, step=2)
    config["model"]["cnn"]["kernel_size"] = [k_size, k_size]

    # --- HPO: MLP Architecture ---
    h1 = trial.suggest_int("nhidden1", 100, 600, step=100)
    h2 = trial.suggest_int("nhidden2", 50, 250, step=50)
    h3 = trial.suggest_int("nhidden3", 10, 100, step=10)
    config["model"]["mlp"]["hidden_layers"] = [h1, h2, h3]
    config["model"]["mlp"]["dropout"] = trial.suggest_float("dropout_rate", 0.1, 0.5)

    # --- HPO: Training Params ---
    config["training"].update({
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_int("batch_size", 16, 128, step=16), # Power of 2 usually better for GPU
        "optimizer": trial.suggest_categorical("optimizer", ["AdamW", "SGD"]) # Changed Adam to AdamW for 2026 SOTA
    })

    # Data Loading
    train_loader = load_train_data(config)
    val_loader = load_val_data(config)

    # Model Building
    if model_name == "CNN_MLP":
        model = CNNMLP(config["model"]["cnn"], config["model"]["mlp"])
    elif model_name == "UNet_A":
        model = UNet_A(config["model"]["cnn"], config["model"]["mlp"])
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Training (Using the refactored loop)
    # val_loss_history is returned but only val_loss is needed for Optuna
    val_loss, _ = train_model_hpo(model, train_loader, val_loader, config, trial)
    
    return val_loss

# ======================================================================
# 3. Execution
# ======================================================================
if __name__ == "__main__":
    # Median pruner is standard for finding MJO signals efficiently
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True
    )

    n_total = 100
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    
    if n_completed >= n_total:
        print(f"[INFO] Study already has {n_completed} trials. Exiting.")
        sys.exit(0)

    n_to_run = n_total - n_completed
    print(f"[INFO] Running {n_to_run} trials on this worker.")

    study.optimize(objective, n_trials=n_to_run)
    
    print(f"Best Trial: {study.best_trial.number}")
    print(f"Best Value: {study.best_trial.value}")

