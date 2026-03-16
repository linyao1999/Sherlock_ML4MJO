import yaml
from datetime import datetime, timedelta
import xarray as xr
import pandas as pd 
import numpy as np  
from torch.utils.data import Dataset, DataLoader
import torch    
import os 

# Global cache to ensure we only calculate mean/std once across all dataset splits
_NORM_STATS_CACHE = {}

class MapsDataset(Dataset):
    def __init__(self, input_path, target_path, date_start, date_end, lead, lat_range=20, transform=None, multi_lead=True):
        assert isinstance(input_path, (list, tuple)), "input_path should be a list/tuple of file paths"

        self.transform = transform
        self.has_logged = False
        self.multi_lead = multi_lead
        self.lead = lead

        # --- NORMALIZATION CACHE LOGIC ---
        global _NORM_STATS_CACHE
        cache_key = str(input_path) # Unique key based on your file paths
        
        if cache_key not in _NORM_STATS_CACHE:
            print("Pre-computing normalization stats (1979-2001)... This will only happen once.")
            stats = []
            for ipath in input_path:
                with xr.open_dataarray(ipath) as da_full:
                    # Calculate stats on the training period
                    da_train = da_full.sel(time=slice('1979-01-01', '2001-12-31'), lat=slice(lat_range, -lat_range))
                    stats.append((da_train.mean().values, da_train.std().values))
            _NORM_STATS_CACHE[cache_key] = stats
        
        # Retrieve the pre-computed stats
        norm_stats = _NORM_STATS_CACHE[cache_key]
        # ---------------------------------

        # 1. Load and normalize inputs
        inputs = []
        for idx, ipath in enumerate(input_path):
            with xr.open_dataarray(ipath) as da_full:
                da = da_full.sel(time=slice(date_start, date_end), lat=slice(lat_range, -lat_range))
                da_mean, da_std = norm_stats[idx]
                inputs.append(((da - da_mean) / da_std).expand_dims('variable'))
        
        input_xr = xr.concat(inputs, dim='variable').transpose('time', 'variable', 'lat', 'lon')
        self.input_data = input_xr.values.astype(np.float32)
        self.times = input_xr.time.values 

        # 2. Handle Time Slicing for Lead Time
        date_start_dt = datetime.strptime(date_start, "%Y-%m-%d")
        date_end_dt = datetime.strptime(date_end, "%Y-%m-%d")
        
        last_input_date = pd.Timestamp(self.times[-1]).to_pydatetime()
        target_end_date = date_end_dt + timedelta(days=lead)

        if target_end_date > last_input_date:
            date_end_dt = last_input_date - timedelta(days=lead)
            new_len = len(pd.date_range(date_start_dt, date_end_dt))
            self.input_data = self.input_data[:new_len]

        # 3. Target Data Preparation
        output_data = []
        with xr.open_dataarray(target_path) as target_da_full:
            if self.multi_lead:
                for le in range(0, lead + 1):
                    t_start = (date_start_dt + timedelta(days=le)).strftime("%Y-%m-%d")
                    t_end = (date_end_dt + timedelta(days=le)).strftime("%Y-%m-%d")
                    output_data.append(target_da_full.sel(time=slice(t_start, t_end)).values)
                
                self.target_data = np.transpose(np.stack(output_data, axis=0), (1, 0, 2)).astype(np.float32)
            else:
                t_start = (date_start_dt + timedelta(days=lead)).strftime("%Y-%m-%d")
                t_end = (date_end_dt + timedelta(days=lead)).strftime("%Y-%m-%d")
                self.target_data = target_da_full.sel(time=slice(t_start, t_end)).values.astype(np.float32)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_tensor = torch.from_numpy(self.input_data[idx])
        target_tensor = torch.from_numpy(self.target_data[idx].flatten())

        if not self.has_logged:
            print(f"Mode: {'Multi' if self.multi_lead else 'Single'}-Lead | "
                  f"Input: {input_tensor.shape} | Target: {target_tensor.shape}")
            self.has_logged = True

        return input_tensor, target_tensor

        
# --- Updated Loaders (Original Names Preserved) ---
# Added pin_memory=True to all loaders to speed up CPU-to-GPU memory transfers.

def load_train_data(config):
    multi_lead = config["training"].get("multi_lead", True)
    train_data = MapsDataset(
        config["data"]["input_path"], config["data"]["target_path"],
        config["data"]["train_start"], config["data"]["train_end"], 
        config["data"]["lead"], config["data"]["lat_range"], 
        config["data"].get("transform"), multi_lead=multi_lead
    )
    return DataLoader(train_data, batch_size=config["training"]["batch_size"], shuffle=True, pin_memory=True)

def load_val_data(config):
    multi_lead = config["training"].get("multi_lead", True)
    val_data = MapsDataset(
        config["data"]["input_path"], config["data"]["target_path"],
        config["data"]["val_start"], config["data"]["val_end"], 
        config["data"]["lead"], config["data"]["lat_range"], 
        config["data"].get("transform"), multi_lead=multi_lead
    )
    return DataLoader(val_data, batch_size=config["training"]["batch_size"], shuffle=False, pin_memory=True)

def load_test_data(config):
    multi_lead = config["training"].get("multi_lead", True)
    test_data = MapsDataset(
        config["data"]["input_path"], config["data"]["target_path"],
        config["data"]["test_start"], config["data"]["test_end"], 
        config["data"]["lead"], config["data"]["lat_range"], 
        config["data"].get("transform"), multi_lead=multi_lead
    )
    return DataLoader(test_data, batch_size=config["training"]["batch_size"], shuffle=False, pin_memory=True)

def get_time_dimension(input_path, date_start, date_end, lead, mem=0):
    # (Unchanged)
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
