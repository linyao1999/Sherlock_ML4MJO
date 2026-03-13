import yaml
from datetime import datetime, timedelta
import xarray as xr
import pandas as pd 
import numpy as np  
from torch.utils.data import Dataset, DataLoader
import torch    

class MapsDataset(Dataset):
    def __init__(self, input_path, target_path, date_start, date_end, lead, lat_range=20, transform=None, multi_lead=True):
        """
        Args:
            lead (int): If multi_lead=True, this is the max lead (0 to lead). 
                        If multi_lead=False, this is the specific target lead day.
            multi_lead (bool): Whether to return a sequence of leads or just the specific lead day.
        """
        assert isinstance(input_path, (list, tuple)), "input_path should be a list/tuple of file paths"

        self.n_var = len(input_path)
        self.transform = transform
        self.has_logged = False
        self.multi_lead = multi_lead
        self.lead = lead

        # 1. Load and normalize inputs
        inputs = []
        for ipath in input_path:
            da = xr.open_dataarray(ipath).sel(time=slice(date_start, date_end), lat=slice(lat_range, -lat_range))
            da_train = xr.open_dataarray(ipath).sel(time=slice('1979-01-01', '2001-12-31'), lat=slice(lat_range, -lat_range))
            da_mean = da_train.mean().values
            da_std = da_train.std().values
            inputs.append(((da - da_mean) / da_std).expand_dims('variable'))
        
        self.input = xr.concat(inputs, dim='variable').transpose('time', 'variable', 'lat', 'lon')

        # 2. Handle Time Slicing for Lead Time
        date_start_dt = datetime.strptime(date_start, "%Y-%m-%d")
        date_end_dt = datetime.strptime(date_end, "%Y-%m-%d")
        
        # We must ensure the last input date + lead exists in the target data
        target_end_date = date_end_dt + timedelta(days=lead)
        last_input_date = pd.Timestamp(self.input.time[-1].values).to_pydatetime()

        if target_end_date > last_input_date:
            target_end_date = last_input_date
            date_end_dt = target_end_date - timedelta(days=lead)
            self.input = self.input.sel(time=slice(date_start, date_end_dt.strftime("%Y-%m-%d")))

        # 3. Target Data Preparation
        output_data = []
        
        if self.multi_lead:
            # Shape: [time, lead+1, n_modes] -> Flattened to [time, (lead+1)*n_modes]
            for le in range(0, lead + 1):
                t_start = date_start_dt + timedelta(days=le)
                t_end = date_end_dt + timedelta(days=le)
                target_da = xr.open_dataarray(target_path).sel(
                    time=slice(t_start.strftime("%Y-%m-%d"), t_end.strftime("%Y-%m-%d"))
                )
                output_data.append(target_da.values)
            
            # Stack and Transpose: [lead+1, time, modes] -> [time, lead+1, modes]
            self.target = np.transpose(np.stack(output_data, axis=0), (1, 0, 2))
        else:
            # Single Lead: Only grab the specific day offset by 'lead'
            t_start = date_start_dt + timedelta(days=lead)
            t_end = date_end_dt + timedelta(days=lead)
            target_da = xr.open_dataarray(target_path).sel(
                time=slice(t_start.strftime("%Y-%m-%d"), t_end.strftime("%Y-%m-%d"))
            )
            # Shape: [time, n_modes]
            self.target = target_da.values

        if len(self.input.time) != len(self.target):
            raise ValueError(f"Dim mismatch: Input {len(self.input.time)} vs Target {len(self.target)}")

    def __len__(self):
        return len(self.input.time)

    def __getitem__(self, idx):
        input_tensor = self.input.isel(time=idx).values
        if input_tensor.ndim == 2:
            input_tensor = np.expand_dims(input_tensor, axis=0)

        # Flatten target: 
        # Multi-lead becomes [(lead+1) * n_modes]
        # Single-lead becomes [n_modes]
        target_tensor = self.target[idx].flatten()

        input_tensor = torch.as_tensor(input_tensor, dtype=torch.float32)
        target_tensor = torch.as_tensor(target_tensor, dtype=torch.float32)

        if not self.has_logged:
            print(f"Mode: {'Multi' if self.multi_lead else 'Single'}-Lead | "
                  f"Input: {input_tensor.shape} | Target: {target_tensor.shape}")
            self.has_logged = True

        return input_tensor, target_tensor

# Update Loaders to pass the 'multi_lead' flag from config
def load_train_data(config):
    multi_lead = config["training"].get("multi_lead", True)
    train_data = MapsDataset(
        config["data"]["input_path"], config["data"]["target_path"],
        config["data"]["train_start"], config["data"]["train_end"], 
        config["data"]["lead"], config["data"]["lat_range"], 
        config["data"].get("transform"), multi_lead=multi_lead
    )
    return DataLoader(train_data, batch_size=config["training"]["batch_size"], shuffle=True)

def load_val_data(config):
    multi_lead = config["training"].get("multi_lead", True)
    val_data = MapsDataset(
        config["data"]["input_path"], config["data"]["target_path"],
        config["data"]["val_start"], config["data"]["val_end"], 
        config["data"]["lead"], config["data"]["lat_range"], 
        config["data"].get("transform"), multi_lead=multi_lead
    )
    return DataLoader(val_data, batch_size=config["training"]["batch_size"], shuffle=False)

def load_test_data(config):
    multi_lead = config["training"].get("multi_lead", True)
    test_data = MapsDataset(
        config["data"]["input_path"], config["data"]["target_path"],
        config["data"]["test_start"], config["data"]["test_end"], 
        config["data"]["lead"], config["data"]["lat_range"], 
        config["data"].get("transform"), multi_lead=multi_lead
    )
    return DataLoader(test_data, batch_size=config["training"]["batch_size"], shuffle=False)

def get_time_dimension(input_path, date_start, date_end, lead, mem=0):
    input_da = xr.open_dataarray(input_path[0]).sel(time=slice(date_start, date_end))
    date_start_dt = datetime.strptime(date_start, "%Y-%m-%d") + timedelta(days=int(mem))
    
    date_end_dt = datetime.strptime(date_end, "%Y-%m-%d")
    target_end_date = date_end_dt + timedelta(days=lead)
    
    actual_last_date = pd.Timestamp(input_da.time[-1].values).to_pydatetime()
    
    if target_end_date > actual_last_date:
        target_end_date = actual_last_date
        date_end_dt = target_end_date - timedelta(days=lead)

    return input_da.sel(
        time=slice(date_start_dt.strftime("%Y-%m-%d"), date_end_dt.strftime("%Y-%m-%d"))
    ).time

