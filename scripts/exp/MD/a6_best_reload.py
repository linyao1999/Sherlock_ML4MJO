import os
import yaml
import sys
from pathlib import Path
import torch

# Add src/ and scripts/ to sys.path
scripts_path = Path(__file__).resolve().parents[2]
sys.path.append(str(scripts_path))
sys.path.append(str(scripts_path / 'src'))

from data_prepare.dataset import load_test_data, get_time_dimension
from models.cnnmlp import CNNMLP
from models.unet import UNet_A
from inference.predict import predict
from utils.save_prediction import save_predictions_with_time
from utils.logger import setup_logger

logger = setup_logger()

'''
Load a specific top-N configuration and its unscaled trained weights,
swap the input dataset to a rescaled version, and generate new predictions.

Environment variables:
  trial_rank      : 1 (must match what was used in a2_best.py)
  exp_num         : 1..16 (must match what was used in a2_best.py)
  dataflg         : dataset flag (e.g., era5)
  unscaled_expflg : the model dir flag (e.g., fltano120)
  rescaled_expflg : the new input dir flag (e.g.,rescaled_m10resi_wnx9resi)
  input_var       : input variable name (e.g., olr)
  output_var      : output variable name (e.g., ROMI)
  model_name      : model name (e.g., UNet_A)

Example:
  trial_rank=1 exp_num=1 unscaled_expflg=unscaled_data rescaled_expflg=rescaled_data python3 a6_best_reload.py
'''

# ======================================================================
# 1. Read Environment Variables
# ======================================================================
exp_num = int(os.environ.get("exp_num", 1))
trial_rank = int(os.environ.get("trial_rank", 1))   

dataflg = os.environ.get("dataflg", "era5").lower()
unscaled_dir = os.environ.get("unscaled_expflg", "fltano120") # model
rescaled_dir = os.environ.get("rescaled_expflg", "rescaled_m10resi_wnx9resi") # new input
input_var_name = os.environ.get("input_var", "olr")
output_var_name = os.environ.get("output_var", "ROMI")
model_name = os.environ.get("model_name", "UNet_A")

# Reconstruct the original experiment name used in a2_best.py to find the config
exp_name_unscaled = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_{unscaled_dir}"
trial_tag = f"t{trial_rank}"

# ======================================================================
# 2. Load the Finalized Config from a2_best.py
# ======================================================================
config_path = f'./yaml/best_config_{exp_name_unscaled}_{trial_tag}.yaml'
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file {config_path} not found. Did you run a2_best.py first?")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# ======================================================================
# 3. Modify Config for Rescaled Data & New Prediction Path
# ======================================================================
logger.info(f"Swapping input paths from '{unscaled_dir}' to '{rescaled_dir}'")

# Swap input paths in the config
new_input_paths = []
for path in config["data"]["input_path"]:
    p = Path(path)
    
    # Rebuild the path by replacing only the exact folder name match
    new_parts = [rescaled_dir if part == unscaled_dir else part for part in p.parts]
    new_path = Path(*new_parts)
    new_input_paths.append(str(new_path))

print('new input:', new_input_paths)
config["data"]["input_path"] = new_input_paths

# Construct new prediction save path so we don't overwrite the unscaled predictions
exp_name_rescaled = f"{dataflg}_{input_var_name}_{model_name}_{output_var_name}_model_{unscaled_dir}_input_{rescaled_dir}"
config["prediction_save_path"] = config["prediction_save_path"].replace(exp_name_unscaled, exp_name_rescaled)
config["prediction_save_path"] = config["prediction_save_path"].replace(f'exp{config["exp_num"]}', f'exp{exp_num}')

prediction_save_path = config["prediction_save_path"]
config["model_save_path"] = config["model_save_path"].replace(f'exp{config["exp_num"]}', f'exp{exp_num}')
model_save_path = config["model_save_path"] # Keep the original model save path to load the weights

# Ensure the config exp_num matches what we are targeting
config["exp_num"] = exp_num

if os.path.exists(prediction_save_path):
    logger.info(f"{prediction_save_path} already exists. Exiting.")
    sys.exit(0)

# =====================================================================
# 4. Build Model & Load Unscaled Weights
# =====================================================================
if config["model"]["name"] == "CNN_MLP":
    model = CNNMLP(config["model"]["cnn"], config["model"]["mlp"])
elif config["model"]["name"] == "UNet_A":
    model = UNet_A(config["model"]["cnn"], config["model"]["mlp"])
else:
    raise ValueError(f"Unsupported model: {config['model']['name']}")

if not os.path.exists(model_save_path):
    raise FileNotFoundError(f"Unscaled model weights not found at {model_save_path}. Ensure a2_best.py saved them.")

logger.info(f"Loading unscaled weights from {model_save_path}...")
model.load_state_dict(torch.load(model_save_path))
model = model.cuda()

# =====================================================================
# 5. Inference / Predictions on Rescaled Data
# =====================================================================
lead = config["data"]["lead"]
logger.info(f"Making predictions on rescaled test data for lead={lead}...")

# Load test data using the updated config pointing to the rescaled dir
test_loader = load_test_data(config)

preds = predict(model, test_loader)
targets = torch.cat([batch[1] for batch in test_loader], dim=0).cpu()

time = get_time_dimension(
    config["data"]["input_path"],
    config["data"]["test_start"],
    config["data"]["test_end"],
    config["data"]["lead"],
)

if len(time) != preds.shape[0] or len(time) != targets.shape[0]:
    raise ValueError(
        f"Dimension mismatch: Time length ({len(time)}) does not match "
        f"predictions ({preds.shape[0]}) or targets ({targets.shape[0]})."
    )

os.makedirs(os.path.dirname(prediction_save_path), exist_ok=True)
save_predictions_with_time(preds, targets, time, prediction_save_path)
logger.info(f"Rescaled predictions saved successfully at {prediction_save_path}")
