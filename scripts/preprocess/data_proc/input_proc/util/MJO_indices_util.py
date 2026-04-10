# This file contains functions to calculate RMM and ROMI index. 
import numpy as np 
import pandas as pd 
import xarray as xr 
import matplotlib.pyplot as plt 
import os 
import logging
import shutil
from datetime import datetime
from scipy.signal import detrend
from scipy.signal import convolve2d
from scipy.stats import ttest_1samp
from scipy import special
import math

# generate daily variable maps
# convert original units to standard units
def rawtodaily_1var(varn, varn_name, data_path, data_store, date_sta, date_end):
    # data_path: path to store the downloaded raw data
    # date_sta: the starting time step
    # date_end: the ending time step

    print('convert to daily starts: ' + varn)
    
    year_sta = int(datetime.strptime(date_sta, '%Y-%m-%d').year)
    year_end = int(datetime.strptime(date_end, '%Y-%m-%d').year)
    
    
    filenames = [data_path + varn +'_2deg' + str(i) + '.nc' for i in range(year_sta, year_end+1)] 

    ds = xr.open_mfdataset(filenames)

    ds1 = ds.resample(time="1D").mean()
    # ds1 = ds1.sel(time=slice(date_sta, date_end))
    ds1 = ds1.rename({'latitude': 'lat', 'longitude':'lon', varn_name: varn})

    if varn == 'olr':
        ds2 = ds1 / - 3600
    else:
        ds2 = ds1 

    ds2.to_netcdf(data_store + varn + '.day.' + str(year_sta) + 'to' + str(year_end) + '.nc', mode='w')

    print('convert to daily ends: ' + varn)

# below are functions to get anomalies (remove smoothed annual cycles; remove previous 120-day averages)
# calculate raw climatological annual cycle. 
# use the periods of 1979-2001
def get_raw_clim_annual_cycle(x, latsel=90, year_sta=1979, year_end=2001):
    # x is a variable 
    # x[time, lat, lon]
    if latsel == 90:
        x_sel = x.sel(time=slice(str(year_sta)+'-01-01', str(year_end)+'-12-31')).fillna(0)
    else:
        x_sel = x.sel(time=slice(str(year_sta)+'-01-01', str(year_end)+'-12-31'), lat=slice(latsel,-latsel)).fillna(0)
    
    x_sel_clim = x_sel.groupby("time.dayofyear").mean()

    x_values = x_sel_clim.values

    print('shape of raw climatological maps: ', x_values.shape)

    x_values[-1,:,:] = 0.5 * (x_values[-2,:,:] + x_values[0,:,:])

    return x_sel_clim, x_values

# calculate the time-mean and the first three harmonics of the annual cycle.
# In this way, we get the smoothed annual cycle.
def get_smoothed_clim_annual_cycle(x):
    # x is raw climotological annual cycle.
    # the shape of x : [366, lat, lon]
    # fourier transform in doy
    x_fft = np.fft.rfft(x, axis=0)  # fourier transform coefficients
    x_fft[4:] = 0.0  # remove high-frequency signals

    x_re = np.fft.irfft(x_fft, x.shape[0], axis=0)  # reconstruct the x using slow-frequency signals

    return x_re

# calculate the anomalies of given daily data. 
def get_anomalies_1var(varn, data_path, date_sta, date_end, latsel=90, yearclim_sta=1979, yearclim_end=2001):
    # the dataset will be cut to get periods between date_sta to date_end, and to get latitudes between -latsel and latsel.
    # use yearclim_sta to yearclim_end to calculate smoothed annual cycle. 
    # read daily data
    print('start compute: ' + varn)

    ds = xr.open_dataset(data_path)
    print('nan values: ', ds[varn].isnull().sum().values)
    
    ds_sel = ds.sel(time=slice(date_sta, date_end), lat=slice(latsel, -latsel)).fillna(0)
    target = ds_sel[varn].copy()  # time_sel, lat_sel, lon

    # remove smoothed annual cycle
    # calculate raw annual cycle
    smoothed_annual, raw_annual = get_raw_clim_annual_cycle(target, latsel=latsel, year_sta=yearclim_sta, year_end=yearclim_end)

    # calculate smoothed annual cycle by reconstracting the annual cycle with the mean and the first 3 harmonics.
    smoothed_annual.values = get_smoothed_clim_annual_cycle(raw_annual)  

    # remove the annual cycle.
    target_ano = target.groupby("time.dayofyear") - smoothed_annual

    return target_ano

def rmv_runavg(x, window_size=120):
    # remove the previous running averages
    xroll = x.rolling(time=window_size).mean()  # previous window-size mean
    xdetrend = (x - xroll).dropna(dim='time')
    return xdetrend
