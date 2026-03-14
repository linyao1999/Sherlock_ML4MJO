# Contains classes and functions for loading and preprocessing data.
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


class MapsDataset(Dataset):
    def __init__(self, input_path, target_path, date_start, date_end, lead, lat_range=20, transform=None):
        """
        Args:
            input_path (list of str): List of paths to input data files (one per variable).
            target_path (str): Path to the target data file.
            date_start (str): Start date in 'YYYY-MM-DD' format.
            date_end (str): End date in 'YYYY-MM-DD' format.
            lead (int): Lead time in days for the target variable.
            lat_range (int): Latitude range to use for the input data.
            transform (callable, optional): Optional transform to apply to the data.
        """
        assert isinstance(input_path, (list, tuple)), "input_path should be a list/tuple of file paths"

        self.n_var = len(input_path)
        self.transform = transform
        self.has_logged = False

        # Load and preprocess each variable, stack into shape [time, n_var, lat, lon]
        inputs = []
        for ipath in input_path:
            da = xr.open_dataarray(ipath).sel(time=slice(date_start, date_end), lat=slice(lat_range, -lat_range))
            # Normalize per variable using the training period (hardcoded here, change as needed)
            da_train = xr.open_dataarray(ipath).sel(time=slice('1979-01-01', '2001-12-31'), lat=slice(lat_range, -lat_range))
            da_mean = da_train.mean().values
            da_std = da_train.std().values
            inputs.append(((da - da_mean) / da_std).expand_dims('variable'))
        # Stack on new 'variable' dimension
        self.input = xr.concat(inputs, dim='variable')  # [variable, time, lat, lon]
        self.input = self.input.transpose('time', 'variable', 'lat', 'lon')  # [time, variable, lat, lon]

        # Target time handling
        date_start_dt = datetime.strptime(date_start, "%Y-%m-%d")
        date_end_dt = datetime.strptime(date_end, "%Y-%m-%d")
        
        target_end_date = date_end_dt + timedelta(days=lead)
        last_input_date = pd.Timestamp(self.input.time[-1].values).to_pydatetime()
        # print(f"Raw input date: {self.input.time[0].values} to {self.input.time[-1].values}")

        if target_end_date > last_input_date:
            target_end_date = last_input_date
            date_end_dt = target_end_date - timedelta(days=lead)
            # Re-slice input so that input and target times match
            self.input = self.input.sel(time=slice(date_start, date_end_dt.strftime("%Y-%m-%d")))
            # print(f"Updated input end date: {date_end_dt.strftime('%Y-%m-%d')}")

        # Target data
        # Prepare target for all leads
        output_allleads = []
        for le in range(0, lead+1):
            target_start = date_start_dt + timedelta(days=le)
            target_end = date_end_dt + timedelta(days=le)
            target_da = xr.open_dataarray(target_path).sel(
                time=slice(target_start.strftime("%Y-%m-%d"), target_end.strftime("%Y-%m-%d"))
            )
            output_allleads.append(target_da.values)
        # [lead+1, time, n_modes]
        output_allleads = np.stack(output_allleads, axis=0)  # shape: [lead+1, time, n_modes]
        output_allleads = np.transpose(output_allleads, (1, 0, 2))  # [time, lead+1, n_modes]
        self.target = output_allleads

        # print(f"Input shape: {self.input.shape}")
        # print(f"Target shape: {self.target.shape}")
        # print(f"Target time from {target_start.strftime('%Y-%m-%d')} to {target_end.strftime('%Y-%m-%d')}")
        # self.target = xr.open_dataarray(target_path).sel(
        #     time=slice(target_start_date.strftime("%Y-%m-%d"), target_end_date.strftime("%Y-%m-%d"))
        # )

        # Ensure lengths match
        if len(self.input.time) != len(self.target):
            raise ValueError(f"Input and target dimensions do not match: {len(self.input.time)} vs {len(self.target)}")

    def __len__(self):
        return len(self.input.time)

    def __getitem__(self, idx):
        # Input shape: [time, n_var, lat, lon]
        input = self.input.isel(time=idx).values   # shape: [n_var, lat, lon]
        if input.ndim == 2:  # [lat, lon], only one variable, add channel dimension
            input = np.expand_dims(input, axis=0)  # Now [1, lat, lon]

        target = self.target[idx].flatten()  # shape: [lead+1, n_modes]

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        # Print shape once for debug
        if not self.has_logged:
            print(f"Input shape: {input.shape}, Target shape: {target.shape}")
            input_time = self.input.time[idx].values
            print(f"Input time: {input_time}")
            self.has_logged = True

        return input, target

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # print the config
    # print(yaml.dump(config))

    return config

def load_train_data(config):
    # Load training and validation data

    train_data = MapsDataset(config["data"]["input_path"], config["data"]["target_path"],
                            config["data"]["train_start"], config["data"]["train_end"], 
                            config["data"]["lead"], config["data"]["lat_range"], config["data"]["transform"])
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=config["training"]["batch_size"], shuffle=True)

    return train_loader

def load_val_data(config):
    # Load training and validation data
    val_data = MapsDataset(config["data"]["input_path"], config["data"]["target_path"],
                          config["data"]["val_start"], config["data"]["val_end"], 
                              config["data"]["lead"], 
                              config["data"]["lat_range"], config["data"]["transform"])
    # Create data loaders
    val_loader = DataLoader(val_data, batch_size=config["training"]["batch_size"], shuffle=False)

    return val_loader

def load_test_data(config):
    # Load training and test data
    test_data = MapsDataset(config["data"]["input_path"], config["data"]["target_path"],
                          config["data"]["test_start"], config["data"]["test_end"], 
                          config["data"]["lead"], 
                              config["data"]["lat_range"], config["data"]["transform"])
    # Create data loaders
    test_loader = DataLoader(test_data, batch_size=config["training"]["batch_size"], shuffle=False)

    return test_loader

# clarify the time dimension
def get_time_dimension(input_path, date_start, date_end, lead, mem=0):
    input_da = xr.open_dataarray(input_path[0]).sel(time=slice(date_start, date_end))

    date_start_date = datetime.strptime(date_start, "%Y-%m-%d") + timedelta(days=int(mem))
    date_end_date = datetime.strptime(date_end, "%Y-%m-%d")
    target_end_date = date_end_date + timedelta(days=lead)
    date_end_date = pd.Timestamp(input_da.time[-1].values).to_pydatetime()
    # redefine the end date if target_end_date is greater than the last date in the dataset
    if target_end_date > date_end_date:
        target_end_date = date_end_date
        date_end_date = target_end_date - timedelta(days=lead)

    return input_da.sel(time=slice(date_start_date.strftime("%Y-%m-%d"), date_end_date.strftime("%Y-%m-%d"))).time

