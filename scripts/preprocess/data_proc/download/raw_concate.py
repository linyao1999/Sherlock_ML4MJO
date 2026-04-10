import numpy as np 
import xarray as xr

import multiprocessing as mp

# salloc --nodes 1 --qos interactive --time 03:10:00 --constraint cpu --account=m3312

old_date_end = '2023-12-31'
new_date_sta = '2024-01-01'

varn19 = [
    "u200", "u500", "u850",  
    "q200", "q500", "q850", 
    "T200", "T500", "T850", 
    "Z200", "Z500", "Z850", 
    "v200", "v500", "v850", 
    "olr", "tcwv", "sst", "prep",
]

varn_name_list = [
    'u', 'u','u',
    'q','q','q',
    't','t','t',
    'z','z','z',
    'v','v','v',
    'ttr','tcwv','sst','tp'
] 

# File paths
template_old = "/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/daily/{}.day.nc"
template_new = "/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/new_raw/{}_2deg_2024to2025.nc"
template_output = "/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/daily/new/{}.day.nc"

def process_variable(var):
    old_file = template_old.format(var)
    new_file = template_new.format(var)
    output_file = template_output.format(var)
    
    # Open datasets
    old_ds = xr.open_dataset(old_file).sel(time=slice(None, old_date_end))
    new_ds = xr.open_dataset(new_file).sel(valid_time=slice(new_date_sta, None))
    new_ds = new_ds.resample(valid_time="1D").mean()
    # change the name of the dimensions and variables to match the old dataset
    new_ds = new_ds.rename({"latitude": "lat", "longitude": "lon", "valid_time": "time", varn_name_list[varn19.index(var)]: var})
    # Drop unnecessary variables if they exist
    for var_to_drop in ["pressure_level", "number", "expver"]:
        if var_to_drop in new_ds:
            new_ds = new_ds.drop_vars(var_to_drop)
    
    # Squeeze the dataset to remove singleton dimensions
    new_ds = new_ds.squeeze()

    if var == "olr":
        new_ds = new_ds / (-3600)
    
    # Concatenate and sort
    combined_ds = xr.concat([old_ds, new_ds], dim="time").sortby("time")
    
    # Save output
    combined_ds.to_netcdf(output_file)
    print(f"Concatenated dataset for {var} saved to {output_file}")

pool = mp.Pool(19)
pool.map(process_variable, varn19)
pool.close()
pool.join()
