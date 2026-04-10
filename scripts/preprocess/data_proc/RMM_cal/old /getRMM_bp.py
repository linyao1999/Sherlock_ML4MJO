'''
This file is to calculate RMM index from flt120d anomaly daily data, use bandpass filter to calculate EOF patterns.
'''
import numpy as np 
import xarray as xr 
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.fft as fft 
import pandas as pd 

import multiprocessing
from util import get_RMMEOF, get_RMM
from util import Lanczos
import os 
# we will get data anomalies for periods of date_sta to date_end
year_sta = '1978'
year_end = '2023'
# this indicates which years are used to get the annual cycle.
yearclim_sta=1979
yearclim_end=2012
eof_lat = 20

fnolr = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/anomaly/olr.anomaly.1978to2023based'+str(yearclim_sta)+'to'+str(yearclim_end)+'.nc'
fnu850 = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/anomaly/u850.anomaly.1978to2023based'+str(yearclim_sta)+'to'+str(yearclim_end)+'.nc'
fnu200 = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/anomaly/u200.anomaly.1978to2023based'+str(yearclim_sta)+'to'+str(yearclim_end)+'.nc'

# filter anomaly files using bandpass filter
# Define the Lanczos filter parameters
filt_type = 'bp'  # Low-pass filter
flg = str(eof_lat) + 'deg' + filt_type
nwts = 201         # Number of weights (must be odd)
pca = 20          # First cut-off period
pcb = 100
delta_t = 1   # Time-step (sampling interval)

# Create the Lanczos filter instance
lanczos_filter = Lanczos(filt_type, nwts, pca, pcb, delta_t=delta_t)

# daily data is stored in this directory.
dirn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/anomaly/'

# anomaly output is store in this directory.
# NOTE: remember to create the directory. 
data_store = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/bpano/'
os.makedirs(data_store, exist_ok=True)

# ================================================


# =========== do not change codes below ================
# variable used 
varn_list = [
    'olr', 'u200', 'u850',
]

data_path_list = [dirn + vari +'.anomaly.1978to2023based'+str(yearclim_sta)+'to'+str(yearclim_end)+'.nc' for vari in varn_list]

# define a function to run get_anomalies_1var
def compute_parallel(varn, data_path):
    target_ano = xr.open_dataset(data_path)[varn]

    filtered_ano = lanczos_filter.wgt_runave(target_ano.values)

    target_ano.values = filtered_ano

    # Assign a name to the DataArray
    data_array = target_ano.rename(varn)

    # Convert the DataArray to a Dataset
    dataset = data_array.to_dataset()

    # Save the Dataset to an NC file
    dataset.to_netcdf(data_store + varn +'.bpano.' + year_sta + 'to' + year_end + 'based' + str(yearclim_sta) + 'to' + str(yearclim_end) + '.nc', mode='w')

    # Close the Dataset
    dataset.close()

pool = multiprocessing.Pool(3)

pool.starmap(compute_parallel, zip(varn_list, data_path_list))

# Close the pool of worker processes
pool.close()
pool.join()

get_RMMEOF(fnolr=data_path_list[0], fnu200=data_path_list[1], fnu850=data_path_list[2], eof_sta=str(yearclim_sta)+'-01-01',eof_end=str(yearclim_end)+'-12-31', eof_lat=eof_lat, flg=flg)
