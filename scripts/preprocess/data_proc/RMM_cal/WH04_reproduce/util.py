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
from eofs.multivariate.standard import MultivariateEof

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
    xroll = x.rolling(time=window_size).mean() # previous window-size mean
    xdetrend = (x - xroll).dropna(dim='time')
    return xdetrend


# Use the same data as WH04 to calculate EOFs
def get_RMMEOF(fnolr, fnu850, fnu200, date_sta=None, date_end=None, eof_sta='1979-01-01', eof_end='2001-12-31', eof_lat=15, flg=''):
    '''
    Input indicates the filenames for filtered anomalies of olr, u850, u200
    eof_sta: the start date used to do EOF analysis
    eof_end: the end date used to do EOF analysis

    reference: https://www.ncl.ucar.edu/Applications/Scripts/mjoclivar_14.ncl

    Steps to calculate RMM from filtered data:
    1. average the filtered data from 15S to 15N
    2. normalize each fields by the square-root of the zonal mean of their temporal variance
    3. concatenate three fields into one
    4. do EOF analysis
    5. normalize PC time series by the standard deviations in 1979-2001

    '''
    # read OLR anomalies
    # 1.
    dsolr = xr.open_dataset(fnolr)
    dsolr = dsolr.sel(lat=slice(eof_lat,-eof_lat), time=slice(date_sta, date_end))
    # averaged over latitude
    avolr = dsolr['olr'].mean(dim="lat")

    # read u850 anomalies
    dsu850 = xr.open_dataset(fnu850)
    dsu850 = dsu850.sel(lat=slice(eof_lat,-eof_lat), time=slice(date_sta, date_end))
    # averaged over latitude
    avu850 = dsu850['u850'].mean(dim="lat")

    # read u200 anomalies
    dsu200 = xr.open_dataset(fnu200)
    dsu200 = dsu200.sel(lat=slice(eof_lat,-eof_lat), time=slice(date_sta, date_end))
    # averaged over latitude
    avu200 = dsu200['u200'].mean(dim="lat")
    # time, lon

    # 2.
    # select training OLR, time x lon
    tmp = avolr.sel(time=slice(eof_sta,eof_end))
    stdolr = tmp.var(dim="time")
    stdolr = stdolr.mean().values
    avolrnm = avolr / (stdolr**(1/2))
    # avolrnm = avolr / stdolr
    print('stdolr: ', stdolr**(1/2))
    del tmp 

    # select training u850, time x lon
    tmp = avu850.sel(time=slice(eof_sta,eof_end))
    stdu850 = tmp.var(dim="time")
    stdu850 = stdu850.mean().values
    avu850nm = avu850 / (stdu850**(1/2))
    # avu850nm = avu850 / stdu850
    print('stdu850: ', stdu850**(1/2))
    del tmp 

    # select training u200, time x lon
    tmp = avu200.sel(time=slice(eof_sta,eof_end))
    stdu200 = tmp.var(dim="time")
    stdu200 = stdu200.mean().values
    avu200nm = avu200 / (stdu200**(1/2))
    # avu200nm = avu200 / stdu200
    print('stdu200: ', stdu200**(1/2))
    del tmp

    # 3. concatenate three fields
    solver = MultivariateEof([np.array(avolrnm.sel(time=slice(eof_sta,eof_end))), np.array(avu850nm.sel(time=slice(eof_sta,eof_end))), np.array(avu200nm.sel(time=slice(eof_sta,eof_end)))], center=True)

    varfrac = solver.varianceFraction()
    print('variance fraction: ', varfrac[:5]*100)

    eof_list = solver.eofs(neofs=2)

    eof1_olr = eof_list[0][0,:]*(-1)
    eof1_u850 = eof_list[1][0,:]*(-1)
    eof1_u200 = eof_list[2][0,:]*(-1)
    # eof1_olr = eof_list[0][0,:]
    # eof1_u850 = eof_list[1][0,:]
    # eof1_u200 = eof_list[2][0,:]

    print('eof1_olr: ', eof1_olr.shape)
    eof1 = np.concatenate((eof1_olr, eof1_u850, eof1_u200), axis=0).squeeze()
    print('eof1: ', eof1.shape)

    eof2_olr = eof_list[0][1,:]
    eof2_u850 = eof_list[1][1,:]
    eof2_u200 = eof_list[2][1,:]
    # eof2_olr = eof_list[0][1,:]*(-1)
    # eof2_u850 = eof_list[1][1,:]*(-1)
    # eof2_u200 = eof_list[2][1,:]*(-1)
    
    eof2 = np.concatenate((eof2_olr, eof2_u850, eof2_u200), axis=0).squeeze()

    eof = np.stack((eof1, eof2), axis=0).T
    print('eof: ', eof.shape)
    lon_coord = np.concatenate([avolr.lon, avolr.lon, avolr.lon])

    EOF_RMM_field = xr.DataArray(eof, dims=['lon', 'mode'], coords={'lon':lon_coord, 'mode':[1,2]})
    EOF_RMM_field1 = EOF_RMM_field
    
    RMM_field = xr.concat([avolrnm, avu850nm, avu200nm], dim='lon')
    # 
    import os 
    if os.path.exists('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/scripts/data_proc/RMM_cal/WH04_reproduce/RMMeof_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc'):
        os.remove('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/scripts/data_proc/RMM_cal/WH04_reproduce/RMMeof_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')

    EOF_RMM_field1.name = 'EOF'
    EOF_RMM_field1.to_netcdf('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/scripts/data_proc/RMM_cal/WH04_reproduce/RMMeof_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')

    if os.path.exists('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/scripts/data_proc/RMM_cal/WH04_reproduce/RMMfield_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc'):
        os.remove('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/scripts/data_proc/RMM_cal/WH04_reproduce/RMMfield_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')

    RMM_field.name = 'RMM_field'
    RMM_field.to_netcdf('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/scripts/data_proc/RMM_cal/WH04_reproduce/RMMfield_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')

def get_RMM(RMM_field, EOF_RMM_field1, eof_sta, eof_end, flg=''):
    
    # PC_RMM_field = RMM_field.dot(EOF_RMM_field1)  # [time, lon] dot [lon, mode] gives [time, mode]
    PC_RMM_field = RMM_field.dot(EOF_RMM_field1)  # [time, lon] dot [lon, mode] gives [time, mode]
    # PC_RMM_field = (RMM_field - RMM_field.sel(time=slice(eof_sta,eof_end)).mean(dim="time")).dot(EOF_RMM_field1)  # [time, lon] dot [lon, mode] gives [time, mode]

    # select the period of 1979-2001 to do normalization 
    tmp = PC_RMM_field.sel(time=slice(eof_sta, eof_end)) # [time, mode]
    # PC_RMM_field1 = (PC_RMM_field - tmp.mean(dim='time')) / tmp.std(dim='time') 
    PC_RMM_field1 = PC_RMM_field / tmp.std(dim='time') 

    print(tmp.std(dim='time'))

    if os.path.exists('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/scripts/data_proc/RMM_cal/WH04_reproduce/RMM_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc'):
        os.remove('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/scripts/data_proc/RMM_cal/WH04_reproduce/RMM_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')
    
    PC_RMM_field1.name = 'RMM'
    PC_RMM_field1.to_netcdf('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/scripts/data_proc/RMM_cal/WH04_reproduce/RMM_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')

def mjo_phase_background(ax, opt=None):
    if opt is not None and "axisExtent" in opt and opt["axisExtent"] > 0:
        axisExtent = opt["axisExtent"]
    else:
        axisExtent = 4

    nPhase = 8

    res = {
        # "gsnDraw": False,
        # "gsnFrame": False,
        "vpXF": 0.1,
        "vpYF": 0.8,
        "trYMinF": -axisExtent,
        "trYMaxF": axisExtent,
        "trXMinF": -axisExtent,
        "trXMaxF": axisExtent + 0.05,
        "vpWidthF": 0.45,
        "vpHeightF": 0.45,
        "tmXBFormat": "f",
        "tmYLFormat": "f",
        "tmXBLabelDeltaF": -0.75,
        "tmYLLabelDeltaF": -0.75,
        "tiXAxisFontHeightF": 0.0167,
        "tiYAxisFontHeightF": 0.0167,
        "tiDeltaF": 1.25,
        "xyLineThicknessF": 1
    }

    rad = 4. * np.arctan(1.0) / 180.
    if opt is not None and "radius" in opt and opt["radius"] > 0:
        radius = opt["radius"]
    else:
        radius = 1.0

    nCirc = 361
    theta = np.linspace(0, 360, nCirc) * rad
    xCirc = radius * np.cos(theta)
    yCirc = radius * np.sin(theta)

    if opt is not None and "tiMainString" in opt:
        res["tiMainString"] = opt["tiMainString"]


    ax.plot(xCirc, yCirc, color="black", linewidth=1.0)

    txres = {
        "fontsize": 8,
        "rotation": 90,
        "va": "center",
        "ha": "center"
    }
    # txid = ax.text(0, 0, "Phase 5 (Maritime) Phase 4", **txres)

    amres = {
        "xytext": (0.5, 0.5),
        "textcoords": "axes fraction",
        "ha": "center",
        "va": "center"
    }
    # ann3 = ax.annotate("Phase 7 (Western Pacific) Phase 6")

    plres = {
        "color": "black",
        "linewidth": 1.0,
        "linestyle": "-",
        "dashes": [8, 4]
    }
    if opt is not None and "gsLineDashPattern" in opt:
        plres["dashes"] = opt["gsLineDashPattern"]

    c45 = radius * np.cos(45 * rad)
    E = axisExtent
    R = radius

    phaLine = np.zeros((nPhase, 4))
    phaLine[0, :] = [R, E, 0, 0]
    phaLine[1, :] = [c45, E, c45, E]
    phaLine[2, :] = [0, 0, R, E]
    phaLine[3, :] = [-c45, -E, c45, E]
    phaLine[4, :] = [-R, -E, 0, 0]
    phaLine[5, :] = [-c45, -E, -c45, -E]
    phaLine[6, :] = [0, 0, -R, -E]
    phaLine[7, :] = [c45, E, -c45, -E]

    for i in range(nPhase):
        ax.plot([phaLine[i, 0], phaLine[i, 1]], [phaLine[i, 2], phaLine[i, 3]], **plres)

    plt.show()


def get_RMM_usingKimsInput(fnolr, fnu850, fnu200, date_sta=None, date_end=None, eof_sta=19790101, eof_end=20011231, eof_lat=15, flg=''):
    '''
    Input indicates the filenames for filtered anomalies of olr, u850, u200
    eof_sta: the start date used to do EOF analysis
    eof_end: the end date used to do EOF analysis

    reference: https://www.ncl.ucar.edu/Applications/Scripts/mjoclivar_14.ncl

    Steps to calculate RMM from filtered data:
    1. average the filtered data from 15S to 15N
    2. normalize each fields by the square-root of the zonal mean of their temporal variance
    3. concatenate three fields into one
    4. do EOF analysis
    5. normalize PC time series by the standard deviations in 1979-2001

    '''
    # read OLR anomalies
    # 1.
    dsolr = xr.open_dataset(fnolr)
    dsolr = dsolr.sel(lat=slice(-eof_lat,eof_lat), time=slice(date_sta, date_end))
    # averaged over latitude
    avolr = dsolr['olr_ano'].mean(dim="lat")

    # read u850 anomalies
    dsu850 = xr.open_dataset(fnu850)
    dsu850 = dsu850.sel(lat=slice(-eof_lat,eof_lat), time=slice(date_sta, date_end))
    # averaged over latitude
    avu850 = dsu850['u850_ano'].mean(dim="lat")

    # read u200 anomalies
    dsu200 = xr.open_dataset(fnu200)
    dsu200 = dsu200.sel(lat=slice(-eof_lat,eof_lat), time=slice(date_sta, date_end))
    # averaged over latitude
    avu200 = dsu200['u200_ano'].mean(dim="lat")
    # time, lon

    # 2.
    # select training OLR, time x lon
    tmp = avolr.sel(time=slice(eof_sta,eof_end))
    stdolr = tmp.var(dim="time")
    stdolr = stdolr.mean().values
    avolrnm = avolr / (stdolr**(1/2))
    del tmp 

    # select training u850, time x lon
    tmp = avu850.sel(time=slice(eof_sta,eof_end))
    stdu850 = tmp.var(dim="time")
    stdu850 = stdu850.mean().values
    avu850nm = avu850 / (stdu850**(1/2))
    del tmp 

    # select training u200, time x lon
    tmp = avu200.sel(time=slice(eof_sta,eof_end))
    stdu200 = tmp.var(dim="time")
    stdu200 = stdu200.mean().values
    avu200nm = avu200 / (stdu200**(1/2))

    # from https://github.com/WillyChap/MJOcast/blob/main/build/lib/MJOcast/utils/ProcessOBS.py
    from eofs.multivariate.standard import MultivariateEof

    solver = MultivariateEof([avolrnm.sel(time=slice(eof_sta,eof_end)).values,avu850nm.sel(time=slice(eof_sta,eof_end)).values,avu200nm.sel(time=slice(eof_sta,eof_end)).values])
    varfrac = solver.varianceFraction()
    print('variance fraction: ', varfrac[:5] * 100)
    RMM_field = xr.concat([avolrnm,avu850nm,avu200nm], dim="lon")

    # Convert list of lists to NumPy arrays for easy manipulation
    EOF_RMM_field = np.concatenate(solver.eofs(neofs=2), axis=-1)  # [2 modes, 3*180 lons]

    # Convert to xarray DataArray with correct dimensions
    EOF_RMM_field = xr.DataArray(
        EOF_RMM_field.T,  # Transpose to match (lon, mode)
        dims=['lon', 'mode'], 
        coords={'mode': np.arange(EOF_RMM_field.shape[0]), 'lon': RMM_field.lon}
    )

    pcs_eof = np.array(solver.pcs(npcs=2)) # [time, mode]
    pcs_scale = np.sqrt(np.var(pcs_eof, axis=0, keepdims=True))
    print("Standard deviation of PCs (pcs_scale):", pcs_scale)

    # Change the Sign of EOF to be consistent with WH04
    ieof1max, ieof2max = EOF_RMM_field[0:180,:].argmax(dim="lon")
    lonmaxeof1 = EOF_RMM_field.lon[ieof1max]
    lonmaxeof2 = EOF_RMM_field.lon[ieof2max]

    if (lonmaxeof1 >= 100) & (lonmaxeof1 <= 160) :
        EOF_RMM_field[:,0] = - EOF_RMM_field[:,0]

    if (lonmaxeof2 >= 120) & (lonmaxeof2 <= 220) :
        EOF_RMM_field[:,1] = - EOF_RMM_field[:,1]

    # EOF_RMM_field, eigenvalue1 = get_EOF1979()

    # project the whole dataset onto the EOF during training
    EOF_RMM_field1 = EOF_RMM_field.copy()

    # 
    import os 
    if os.path.exists('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMMeof_KIM_daily_1979to2001.nc'):
        os.remove('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMMeof_KIM_daily_1979to2001.nc')

    EOF_RMM_field1.name = 'EOF'
    EOF_RMM_field1.to_netcdf('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMMeof_KIM_daily_1979to2001.nc')

    # calculate RMM using the EOF
    PC_RMM_field = RMM_field.dot(EOF_RMM_field1)  # [time, lon] dot [lon, mode] gives [time, mode]

    PC_RMM_field1 = PC_RMM_field / pcs_scale

    if os.path.exists('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMM_KIM_daily_1979to2001.nc'):
        os.remove('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMM_KIM_daily_1979to2001.nc')
    
    PC_RMM_field1.name = 'RMM'
    PC_RMM_field1.to_netcdf('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMM_KIM_daily_1979to2001.nc')
