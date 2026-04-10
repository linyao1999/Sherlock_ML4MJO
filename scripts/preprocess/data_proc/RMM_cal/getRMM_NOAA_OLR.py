'''
This file is to calculate RMM index from flt120d anomaly daily data
but for NOAA OLR data, ERA5 u200 and u850 data
'''
import numpy as np 
import xarray as xr 
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.fft as fft 
import pandas as pd 

import multiprocessing
from util import get_RMMEOF, get_RMM

import os 

# salloc --nodes 1 --qos interactive --time 03:10:00 --constraint cpu --account=m3312


# we will get data anomalies for periods of date_sta to date_end
year_sta = '1980'
year_end = '2022'
# this indicates which years are used to get the annual cycle.
yearclim_sta=1980
yearclim_end=2001
eof_lat = 15
flg = '.noaa'

fnolr = f'/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/fltano120/olr.fltano120.noaa.2x2.1979to2022based1979to2001.nc'
fnu850 = f'/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/fltano120/u850.fltano120.1978to2025based1979to2001.nc'
fnu200 = f'/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/fltano120/u200.fltano120.1978to2025based1979to2001.nc'

# anomaly output is store in this directory.
# NOTE: remember to create the directory. 
data_store = '/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/'
os.makedirs(data_store, exist_ok=True)

# =========== do not change codes below ================

get_RMMEOF(fnolr=fnolr, fnu200=fnu200, fnu850=fnu850, eof_sta=str(yearclim_sta)+'-01-01',eof_end=str(yearclim_end)+'-12-31', eof_lat=eof_lat, flg=flg)

dseof = xr.open_dataset(data_store+'RMMeof_ERA5_daily_'+str(yearclim_sta)+'to'+str(yearclim_end)+flg+'.nc')
EOF_RMM_field1 = dseof['EOF']

ds = xr.open_dataset(data_store+'RMMfield_ERA5_daily_'+str(yearclim_sta)+'to'+str(yearclim_end)+flg+'.nc')
RMM_field = ds['RMM_field']

get_RMM(RMM_field, EOF_RMM_field1, str(yearclim_sta)+'-01-01',str(yearclim_end)+'-12-31', flg=flg)