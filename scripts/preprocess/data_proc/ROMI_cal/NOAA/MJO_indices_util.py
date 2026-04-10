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
    ds_sel = ds.sel(time=slice(date_sta, date_end), lat=slice(latsel, -latsel)).fillna(0)
    target = ds_sel[varn]  # time_sel, lat_sel, lon

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

# calculate the Wheeler and Kiladis plot (wavenumber v.s. frequency) for input maps
# reference: https://journals.ametsoc.org/view/journals/atsc/56/3/1520-0469_1999_056_0374_ccewao_2.0.co_2.xml

# decompose the input map into symmetric and antisymmetric parts.
def decompose2SymAsym(arr):
    # copy from https://github.com/brianpm/wavenumber_frequency/blob/master/wavenumber_frequency_functions.py
    """Mimic NCL function to decompose into symmetric and asymmetric parts.
    arr: xarra DataArray
    return: symmetric and asymmetric parts. 
    """
    lat_dim = arr.dims.index('lat')
    # print('decompose along axis=', str(lat_dim))
    data_sym = 0.5*(arr.values + np.flip(arr.values, axis=lat_dim))
    data_asy = 0.5*(arr.values - np.flip(arr.values, axis=lat_dim))
    data_sym = xr.DataArray(data_sym, dims=arr.dims, coords=arr.coords, name='sym')
    data_asy = xr.DataArray(data_asy, dims=arr.dims, coords=arr.coords, name='Asym')

    out = arr.copy()  # might not be best to copy, but is safe        
    out.loc[{'lat':arr['lat'][arr['lat']<0]}] = data_sym.isel(lat=data_sym.lat<0)
    out.loc[{'lat':arr['lat'][arr['lat']>0]}] = data_asy.isel(lat=data_asy.lat>0)
    return out

def split_hann_taper(seg_size, fraction):
    '''
    seg_size: the size of the taper;
    fraction: the fraction of the total points used to do hanning window
    '''
    npts = int(np.rint(seg_size * fraction))  # the total length of hanning window
    hann_taper = np.hanning(npts)  # generate Hanning taper
    taper = np.ones(seg_size)  # create the split cosine bell taper

    # copy the first half of hanner taper to target taper
    taper[:npts//2] = hann_taper[:npts//2]
    # copy the second half of hanner taper to the target taper
    # taper[-npts//2-1:] = hann_taper[-npts//2-1:]
    taper[-(npts//2):] = hann_taper[-(npts//2):]
    return taper

def Hayashi(varfft, nday):
    # use Hayashi method to reorder wavenumber-frequency matrix
    # For ffts that return the coefficients as described above, here is the algorithm
    # coeff array varfft(...,n,t)   dimensioned (...,0:numlon-1,0:numtim-1)
    # new space/time pee(...,pn,pt) dimensioned (...,0:numlon  ,0:numtim  ) 
    #
    # NOTE: one larger in both freq/space dims
    # copied from ncl script
    #
    #    if  |  0 <= pn <= numlon/2-1    then    | numlon/2 <= n <= 1
    #        |  0 <= pt < numtim/2-1             | numtim/2 <= t <= numtim-1
    #
    #    if  |  0         <= pn <= numlon/2-1    then    | numlon/2 <= n <= 1
    #        |  numtime/2 <= pt <= numtim                | 0        <= t <= numtim/2
    #
    #    if  |  numlon/2  <= pn <= numlon    then    | 0  <= n <= numlon/2
    #        |  0         <= pt <= numtim/2          | numtim/2 <= t <= 0
    #
    #    if  |  numlon/2   <= pn <= numlon    then    | 0        <= n <= numlon/2
    #        |  numtim/2+1 <= pt <= numtim            | numtim-1 <= t <= numtim/2
    mlon = len(varfft['wavenumber'])
    mtim = len(varfft['frequency'])

    M = ((mlon - 1)//2) * 2 + 1 
    N = ((mtim - 1)//2) * 2 + 1 

    varspacetime = np.empty((varfft.shape[0],varfft.shape[1], M, N), dtype=varfft.dtype)

    varspacetime[:, :, 0:((mlon-1)//2), 0:((mtim-1)//2) ] = varfft[:, :, ((mlon-1)//2):0:-1, -((mtim-1)//2): ]  
    varspacetime[:, :, 0:((mlon-1)//2), ((mtim-1)//2): ] = varfft[:, :, ((mlon-1)//2):0:-1, 0:((mtim-1)//2 + 1) ]  
    varspacetime[:, :, ((mlon-1)//2):, 0:((mtim-1)//2 + 1) ] = varfft[:, :, 0:((mlon-1)//2 + 1), ((mtim-1)//2)::-1 ]  
    varspacetime[:, :, ((mlon-1)//2):, ((mtim-1)//2 + 1): ] = varfft[:, :, 0:((mlon-1)//2 + 1), -1:(-((mtim-1)//2) - 1):-1]  

    # print('test')
    pee = np.absolute(varspacetime)**2
    # print('test1')
    wave = np.arange(-((mlon - 1)//2), ((mlon - 1)//2 + 1), 1, dtype=int)
    freq = np.linspace((- ((mtim-1)//2)/nday), ((mtim-1)//2)/nday, N)
    
    out = xr.DataArray(
        data=pee,
        dims=("time","lat","wavenumber","frequency"),
        coords={
            "time": varfft["time"],
            "lat": varfft["lat"],
            "wavenumber": wave,
            "frequency": freq,
        }
    )

    return out

def spacetime_power(data, segsize=96, noverlap=60, spd=1, lat_lim=15):
    """
    Perform space-time spectral decomposition and return raw power spectrum following Wheeler-Kiladis approach.

    data: an xarray DataArray to be analyzed; needs to have (time, lat, lon) dimensions.
    segsize: integer denoting the size of time samples that will be decomposed (typically about 96)
    noverlap: integer denoting the number of days of overlap from one segment to the next
    spd: sampling rate, in "samples per day" (e.g. daily=1, 6-houry=4)
    latitude_bounds: a tuple of (southern_extent, northern_extent) to reduce data size.

    Method
    ------------------
        1. Subsample in latitude if latitude_bounds is specified.
        2. Construct symmetric/antisymmetric array .
        3. Construct overlapping window view of data.
        4. Detrend the segments (remove linear trend).
        5. Apply taper in time dimension of windows (aka segments).
        6. Fourier transform
        7. calculate the power.
        8. average over all segments
        9. sum the power over all latitudes.
        
    Notes
    ---------------------------
        Upon returning power, this should be comparable to "raw" spectra. 
        Next step would be be to smooth with `smooth_wavefreq`, 
        and divide raw spectra by smooth background to obtain "significant" spectral power.
        
    """

    segsize = spd * segsize  # how many time steps included.
    noverlap = spd * noverlap  

    # 1.
    data1 = data.sel(lat=slice(lat_lim, -lat_lim))
    # 2. 
    data_sym_asym = decompose2SymAsym(data1)
    # 3. 
    x_roll_sym0 = data_sym_asym.rolling(time=segsize, min_periods=segsize).construct("segments").dropna('time')  # WK99 use 96-day window
    # x_roll_asym0 = data_asym.rolling(time=segsize, min_periods=segsize).construct("segments").dropna('time')  # WK99 use 96-day window
    
    # set overlaps
    x_roll_sym = x_roll_sym0.isel(time=slice(None, None, segsize-noverlap))
    # x_roll_asym = x_roll_asym0.isel(time=slice(None, None, segsize-noverlap))

    seg_dim = x_roll_sym.dims.index('segments')
    # print('seg_dim: ' + str(seg_dim))

    # 4. 
    # Apply the detrend function to each segment
    x_detrend_sym = xr.apply_ufunc(
        detrend,
        x_roll_sym,
        kwargs={'axis': seg_dim},
        dask='parallelized',
        output_dtypes=[x_roll_sym.dtype]
    )
    # x_detrend_asym = xr.apply_ufunc(
    #     detrend,
    #     x_roll_asym,
    #     kwargs={'axis': seg_dim},
    #     dask='parallelized',
    #     output_dtypes=[x_roll_asym.dtype]
    # )

    # 5. add taper
    # using hanning window w[n] = 0.5 * (1 - cos(2*pi*n/(M-1))) to create split cosine bell taper
    taper = split_hann_taper(seg_size=segsize, fraction=0.5)

    x_detrend_symtap = x_detrend_sym * taper  # time, lat, lon, segments
    # x_detrend_asymtap = x_detrend_asym * taper

    # 6. fourier transform within 2 steps 
    # perform fft in longitude
    lon_dim = x_detrend_symtap.dims.index('lon')
    lon_size = x_detrend_symtap.shape[lon_dim]
    # following ncl script, we normalize the fft coefficients with lon_size
    fft_lon_sym = np.fft.fft(x_detrend_symtap, axis=lon_dim) / lon_size # time, lat, wavenumber, segments
    # fft_lon_asym = np.fft.fft(x_detrend_asymtap, axis=lon_dim) / lon_size # time, lat, wavenumber, segments
    # perform fft in segments
    seg_dim = x_detrend_symtap.dims.index('segments')
    seg_size = x_detrend_symtap.shape[seg_dim]
    fft_lonseg_sym = np.fft.fft(fft_lon_sym, axis=seg_dim) / seg_size  # time, lat, wavenumber, frequency
    # fft_lonseg_asym = np.fft.fft(fft_lon_asym, axis=seg_dim) / seg_size  # time, lat, wavenumber, frequency

    fft_sym = xr.DataArray(
        data=fft_lonseg_sym,
        dims=("time","lat","wavenumber","frequency"),
        coords={
            "time": x_detrend_symtap["time"],
            "lat": x_detrend_symtap["lat"],
            "wavenumber": np.fft.fftfreq(lon_size, 1/lon_size),  # how many cycles along the equator 
            "frequency": np.fft.fftfreq(seg_size, 1/spd),  # the frequency (day-1)
        }
    )
    # fft_asym = xr.DataArray(
    #     data=fft_lonseg_asym,
    #     dims=("time","lat","wavenumber","frequency"),
    #     coords={
    #         "time": x_detrend_symtap["time"],
    #         "lat": x_detrend_symtap["lat"],
    #         "wavenumber": np.fft.fftfreq(lon_size, 1/lon_size),  # how many cycles along the equator 
    #         "frequency": np.fft.fftfreq(seg_size, 1/spd),  # the frequency (day-1)
    #     }
    # )

    # 7. 
    # reorder coef matrix according to ncl script
    fft_reorder = Hayashi(fft_sym, segsize/spd)
    # fft_asym_reorder = Hayashi(fft_asym, segsize/spd)
    
    # 8. average over all segments [wavenumber, frequency]
    zsym = 2.0 * fft_reorder.isel(lat=fft_reorder.lat<0).mean(dim='time').sum(dim='lat').squeeze()
    zsym.name = "power"
    zasym = 2.0 * fft_reorder.isel(lat=fft_reorder.lat>0).mean(dim='time').sum(dim='lat').squeeze()
    zasym.name = "power"

    # get power spectra for each segment for significance test [time, wavenumber, frequency]
    zsym1 = 2.0 * fft_reorder.isel(lat=fft_reorder.lat<0).sum(dim='lat').squeeze()
    zasym1 = 2.0 * fft_reorder.isel(lat=fft_reorder.lat>0).sum(dim='lat').squeeze()

    return zsym.where(zsym['frequency']>0, drop=True), zasym.where(zasym['frequency']>0, drop=True), zsym1.where(zsym1['frequency']>0, drop=True), zasym1.where(zasym1['frequency']>0, drop=True)

def power_avg(sym, asym):
    # average over symmetric and antisymmetric components
    return (sym + asym) * 0.5

def wk_smooth121(x):
    # x is a 1-d numpy array
    # create running average kernal
    kern = np.asarray([1/4,1/2,1/4])
    x1 = np.concatenate(([x[0]], x, [x[-1]]))
    x2 = np.convolve(x1, kern, mode='valid')
    return x2

def power_bag(zavg):
    '''
    zavg: averaged power spectrum [wavenuber, frequency]
    1. smooth wavenumbers
    fq < 0.1, smooth 5 times
    fq < 0.2, smooth 10 times
    fq < 0.3, smooth 20 times
    fq >=0.3, smooth 40 times

    2. smooth positive frequency up to 0.8 cpd (max) 10 times
    '''

    x = zavg.where(zavg['frequency']>0, drop=True) # [wavenumber, positive freq]
    
    fq = x['frequency']
    wn = x['wavenumber']

    # Smooth wavenumbers
    smoothed_x = np.copy(x.values)

    for i in range(len(fq)):
        if fq[i] < 0.1:
            for _ in range(5):
                smoothed_x[:, i] = wk_smooth121(smoothed_x[:, i])
        elif fq[i] < 0.2:
            for _ in range(10):
                smoothed_x[:, i] = wk_smooth121(smoothed_x[:, i])
        elif fq[i] < 0.3:
            for _ in range(20):
                smoothed_x[:, i] = wk_smooth121(smoothed_x[:, i])
        elif fq[i] >= 0.3:
            for _ in range(40):
                smoothed_x[:, i] = wk_smooth121(smoothed_x[:, i])

    # Smooth positive frequency up to 0.8 cpd (max) 10 times
    # pt8cpd = min(np.where(fq >= 0.8)[0])
    
    for i in range(len(wn)):
        for _ in range(10):
            smoothed_x[i, :] = wk_smooth121(smoothed_x[i, :])

    x.values = smoothed_x

    return x

def wk_analysis(x, **kwargs):
    '''
    x is the data to do WK analysis
    optional kwargs:
    segsize, noverlap, spd, lat_lim
    '''

    # get the raw space-time power spectra for symmetric and anti-symmetric components
    # negative frequency has been removed. 
    sym, asym, sym_segs, asym_segs = spacetime_power(x, **kwargs)  # [wavenumber, frequency]

    # average between symmteric and anti-symmetric components for background calculation
    zavg = power_avg(sym, asym)

    # following the ncl scripts, we smooth raw power spectra a little bit
    smooth_sym = sym.copy()
    smooth_asym = asym.copy()
    # smooth along frequency
    wn = sym['wavenumber']
    for i in range(len(wn)):
        smooth_sym[i, :].values = wk_smooth121(sym[i,:].values)
        smooth_asym[i, :].values = wk_smooth121(asym[i,:].values)

    # # remove spurious power (frequency=0)
    # sym.loc[{'frequency':0}] = np.nan
    # asym.loc[{'frequency':0}] = np.nan

    # get the background based on the component average zavg
    background = power_bag(zavg)    

    # normalize using background
    sym_norm = smooth_sym / background
    asym_norm = smooth_asym / background

    # test the significance
    # H0: sym_sges.mean('time') = background
    # sym_sges [time, wavenumber, frequency]
    # background [wavenumber, frequency]

    # sample mean: sym, asym
    # population mean: background
    # # sample standard deviation: 
    # sym_std = sym_segs.std(dim='time').squeeze()
    # asym_std = asym_segs.std(dim='time').squeeze()
    # # number of observations:
    # n = len(sym_segs['time'])

    # sym_t_score = (sym - background) / sym_std
    # calculate the t score
    print(sym_segs.shape)
    print(background.shape)
    tscore_sym, pvalue_sym = ttest_1samp(sym_segs.values, np.reshape(background.values, (1, background.shape[0], background.shape[1])), axis=0, alternative='greater')
    p_sym = np.copy(pvalue_sym)
    p_sym[pvalue_sym < 0.01] = 1.0 # if pvalue is < 0.01, we reject H0 and accept alternative hypothesis.
    p_sym[pvalue_sym >= 0.01] = np.nan

    tscore_asym, pvalue_asym = ttest_1samp(asym_segs.values, np.reshape(background.values, (1, background.shape[0], background.shape[1])), axis=0, alternative='greater')
    p_asym = np.copy(pvalue_asym)
    p_asym[pvalue_asym < 0.01] = 1.0 # if pvalue is < 0.01, we reject H0 and accept alternative hypothesis.
    p_asym[pvalue_asym >= 0.01] = np.nan

    return smooth_sym, smooth_asym, background, sym_norm, asym_norm, sym_norm*p_sym, asym_norm*p_asym

def wk_plot(sym, asym, flg, tlt, savflg=False):

    # Assuming smooth_sym and smooth_asym are DataArrays with dimensions (wavenumber, frequency)
    wavenumber = sym['wavenumber']
    frequency = sym['frequency']

    plt.rcParams.update({'font.size': 18})

    # Create a figure and subplots
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.05], width_ratios=[1, 1])

    # Plot smooth_sym for each frequency
    ax = fig.add_subplot(gs[0, 0])
    if flg == 'norm':
        c = ax.contourf(wavenumber, frequency, sym.T, cmap='Reds', levels=10)
        ax.contour(wavenumber,frequency, sym.T, levels=[1.0])
    else:
        c = ax.contourf(wavenumber, frequency, np.log10(sym.T), cmap='Reds', levels=10)
    
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Frequency')
    ax.set_title(tlt[0])
    ax.set_xlim(-15, 15)
    ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.4])
    # Customize tick labels
    ax.tick_params(axis='both', labelsize=12)

    cax = fig.add_subplot(gs[1, 0])
    cbar = plt.colorbar(c, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)

    # Create a twin axis for the second y-axis
    ax2 = ax.twinx()
    ax2.set_ylabel('Period (days)')

    # Modify tick labels for the second y-axis
    ax2.set_ylim(ax.get_ylim())  # Match the y-axis limits with the first y-axis
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels([f'{1/freq:.1f}' if freq != 0 else '0' for freq in ax.get_yticks()])
    # Customize tick labels
    ax2.tick_params(axis='both', labelsize=12)

    ax = fig.add_subplot(gs[0, 1])
    if flg == 'norm':
        c = ax.contourf(wavenumber, frequency, asym.T, cmap='Reds', levels=10)
    else:
        c = ax.contourf(wavenumber, frequency, np.log10(asym.T), cmap='Reds', levels=10)
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Frequency')
    ax.set_title(tlt[1])
    ax.set_xlim(-15, 15)
    ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.4])
    ax.tick_params(axis='both', labelsize=12)

    cax = fig.add_subplot(gs[1, 1])
    cbar = plt.colorbar(c, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)

    # Create a twin axis for the second y-axis
    ax2 = ax.twinx()
    ax2.set_ylabel('Period (days)')

    # Modify tick labels for the second y-axis
    ax2.set_ylim(ax.get_ylim())  # Match the y-axis limits with the first y-axis
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels([f'{1/freq:.1f}' if freq != 0 else '0' for freq in ax.get_yticks()])
    ax2.tick_params(axis='both', labelsize=12)

    # Adjust the spacing between subplots and colorbars
    plt.subplots_adjust(wspace=0.4, hspace=0.3)

    # Save the figure if savflg is True
    if savflg:
        plt.savefig(flg+'power.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

def wk_plot_significance(sym, asym, symsig, asymsig, tlt, savflg=False):

    # Assuming smooth_sym and smooth_asym are DataArrays with dimensions (wavenumber, frequency)
    wavenumber = sym['wavenumber']
    frequency = sym['frequency']

    plt.rcParams.update({'font.size': 18})

    # Create a figure and subplots
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.05], width_ratios=[1, 1])

    # Plot smooth_sym for each frequency
    ax = fig.add_subplot(gs[0, 0])
    c0 = ax.contour(wavenumber, frequency, sym.T, colors='black')
    ax.clabel(c0, inline=True, fontsize=10)
    c = ax.contourf(wavenumber, frequency, symsig.T, cmap='Reds', levels=10)

    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Frequency')
    ax.set_title(tlt[0])
    ax.set_xlim(-15, 15)
    ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.4])
    # Customize tick labels
    ax.tick_params(axis='both', labelsize=12)

    cax = fig.add_subplot(gs[1, 0])
    cbar = plt.colorbar(c, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)

    # Create a twin axis for the second y-axis
    ax2 = ax.twinx()
    ax2.set_ylabel('Period (days)')

    # Modify tick labels for the second y-axis
    ax2.set_ylim(ax.get_ylim())  # Match the y-axis limits with the first y-axis
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels([f'{1/freq:.1f}' if freq != 0 else '0' for freq in ax.get_yticks()])
    # Customize tick labels
    ax2.tick_params(axis='both', labelsize=12)

    ax = fig.add_subplot(gs[0, 1])
    c0 = ax.contour(wavenumber, frequency, asym.T, colors='black')
    ax.clabel(c0, inline=True, fontsize=10)
    c = ax.contourf(wavenumber, frequency, asymsig.T, cmap='Reds', levels=10)

    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Frequency')
    ax.set_title(tlt[1])
    ax.set_xlim(-15, 15)
    ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.4])
    ax.tick_params(axis='both', labelsize=12)

    cax = fig.add_subplot(gs[1, 1])
    cbar = plt.colorbar(c, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)

    # Create a twin axis for the second y-axis
    ax2 = ax.twinx()
    ax2.set_ylabel('Period (days)')

    # Modify tick labels for the second y-axis
    ax2.set_ylim(ax.get_ylim())  # Match the y-axis limits with the first y-axis
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels([f'{1/freq:.1f}' if freq != 0 else '0' for freq in ax.get_yticks()])
    ax2.tick_params(axis='both', labelsize=12)

    # Adjust the spacing between subplots and colorbars
    plt.subplots_adjust(wspace=0.4, hspace=0.3)

    # Save the figure if savflg is True
    if savflg:
        plt.savefig('normpower_sig.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

def wk_plot_bag(bag, savflg=False):
    # Assuming smooth_sym and smooth_asym are DataArrays with dimensions (wavenumber, frequency)
    wavenumber = bag['wavenumber']
    frequency = bag['frequency']

    plt.rcParams.update({'font.size': 18})

    # Create a figure and subplots
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.05], width_ratios=[1, 1])

    # Plot smooth_sym for each frequency
    ax = fig.add_subplot(gs[0, 0])
    c = ax.contourf(wavenumber, frequency, np.log10(bag.T), cmap='Reds', levels=10)
    
    ax.set_xlabel('Wavenumber')
    ax.set_ylabel('Frequency')
    ax.set_title('log(Background)')
    ax.set_xlim(-15, 15)
    ax.set_yticks([0.05, 0.1, 0.2, 0.3, 0.4])
    # Customize tick labels
    ax.tick_params(axis='both', labelsize=12)

    cax = fig.add_subplot(gs[1, 0])
    cbar = plt.colorbar(c, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=12)

    # Create a twin axis for the second y-axis
    ax2 = ax.twinx()
    ax2.set_ylabel('Period (days)')

    # Modify tick labels for the second y-axis
    ax2.set_ylim(ax.get_ylim())  # Match the y-axis limits with the first y-axis
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels([f'{1/freq:.1f}' if freq != 0 else '0' for freq in ax.get_yticks()])
    # Customize tick labels
    ax2.tick_params(axis='both', labelsize=12)

    # Adjust the spacing between subplots and colorbars
    plt.subplots_adjust(hspace=0.3)

    # Save the figure if savflg is True
    if savflg:
        plt.savefig('background.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

# This section is to calculate RMM index

def get_RMMEOF(fnolr, fnu850, fnu200, date_sta=None, date_end=None, eof_sta='1979-01-01', eof_end='2001-12-31', eof_lat=15, flg=''):
    '''
    Input indicates the filenames for filtered anomalies of olr, u850, u200
    eof_sta: the start date used to do EOF analysis
    eof_end: the end date used to do EOF analysis

    reference: https://www.ncl.ucar.edu/Applications/Scripts/mjoclivar_13.ncl

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

    # 3. concatenate three fields
    RMM_field = xr.concat([avolrnm,avu850nm,avu200nm], dim="lon")

    print('RMM_field shape: ', RMM_field.shape)

    from eofs.xarray import Eof  
    solver = Eof(RMM_field.sel(time=slice(eof_sta,eof_end)), center=True)
    EOF_RMM_field = solver.eofs(neofs=2)  # [mode, lon]
    EOF_RMM_field = EOF_RMM_field.transpose()  # [lon, mode]

    # eigenvalue1 = solver.eigenvalues(neigs=2)

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
    if os.path.exists('/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/RMMeof_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc'):
        os.remove('/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/RMMeof_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')

    EOF_RMM_field1.name = 'EOF'
    EOF_RMM_field1.to_netcdf('/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/RMMeof_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')

    if os.path.exists('/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/RMMfield_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc'):
        os.remove('/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/RMMfield_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')

    RMM_field.name = 'RMM_field'
    RMM_field.to_netcdf('/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/RMMfield_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')

def get_RMM(RMM_field, EOF_RMM_field1, eof_sta, eof_end, flg=''):
    
    # PC_RMM_field = RMM_field.dot(EOF_RMM_field1)  # [time, lon] dot [lon, mode] gives [time, mode]
    PC_RMM_field = (RMM_field - RMM_field.sel(time=slice(eof_sta,eof_end)).mean(dim="time")).dot(EOF_RMM_field1)  # [time, lon] dot [lon, mode] gives [time, mode]

    # select the period of 1979-2001 to do normalization 
    tmp = PC_RMM_field.sel(time=slice(eof_sta, eof_end)) # [time, mode]
    PC_RMM_field1 = (PC_RMM_field - tmp.mean(dim='time')) / tmp.std(dim='time') 
    print(tmp.mean(dim='time'))

    if os.path.exists('/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/RMM_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc'):
        os.remove('/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/RMM_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')
    
    PC_RMM_field1.name = 'RMM'
    PC_RMM_field1.to_netcdf('/pscratch/sd/l/linyaoly/ERA5/reanalysis/rmm/RMM_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')

# lowpass, highpass, bandpass filters 
# time dimension must be the first dimension
class Lanczos(object):


    """
    Class for Lanczos filtering. Inspired from 
    NCL's `filwgts_lanczos <http://www.ncl.ucar.edu/Document/Functions/Built-in/filwgts_lanczos.shtml>`_ and `wgt_runave <http://www.ncl.ucar.edu/Document/Functions/Built-in/wgt_runave.shtml>`_ functions.

    :param str filt_type: The type of filter ("lp"=Low Pass, "hp"=High Pass,
     "bp"=Band Pass
    :param int nwts: Number of weights (must be an odd number)
    :param float pca: First cut-off period
    :param float pcb: Second cut-off period (only for band-pass filters)
    :param float delta_t: Time-step

    """

    def __init__(self, filt_type, nwts, pca, pcb=None, delta_t=1):

        """ Initialisation of the filter """

        self.filt_type = filt_type
        self.nwts = nwts
        self.pca = pca
        self.pcb = pcb
        self.delta_t = delta_t

        if self.nwts % 2 == 0:
            raise IOError('Number of weigths must be odd')

        # Because w(n)=w(-n)=0, we would have only nwts-2
        # effective weight. So we add to weights so as to get rid off that
        nwts = self.nwts+2
        weights = np.zeros([nwts])
        nbwgt2 = nwts // 2

        if self.filt_type == 'lp':

            cutoff = float(self.pca)
            cutoff = self.delta_t/cutoff

            weights[nbwgt2] = 2 * cutoff
            k = np.arange(1., nbwgt2)
            sigma = np.sin(np.pi * k / nbwgt2) * nbwgt2 / (np.pi * k)
            firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
            weights[nbwgt2-1:0:-1] = firstfactor * sigma
            weights[nbwgt2+1:-1] = firstfactor * sigma

        elif self.filt_type == 'hp':

            cutoff = float(self.pca)
            cutoff = self.delta_t/cutoff

            weights[nbwgt2] = 1-2 * cutoff
            k = np.arange(1., nbwgt2)
            sigma = np.sin(np.pi * k / nbwgt2) * nbwgt2 / (np.pi * k)
            firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
            weights[nbwgt2-1:0:-1] = -firstfactor * sigma
            weights[nbwgt2+1:-1] = -firstfactor * sigma

        elif self.filt_type == 'bp':

            cutoff1 = np.max([float(self.pca), float(self.pcb)])
            cutoff1 = self.delta_t/cutoff1

            cutoff2 = np.min([float(self.pca), float(self.pcb)])
            cutoff2 = self.delta_t/cutoff2

            weights[nbwgt2] = 2*cutoff2-2*cutoff1
            k = np.arange(1., nbwgt2)
            sigma = np.sin(np.pi * k / nbwgt2) * nbwgt2 / (np.pi * k)
            firstfactor = (np.sin(2.*np.pi*cutoff1*k)/(np.pi*k)) \
                - (np.sin(2.*np.pi*cutoff2*k)/(np.pi*k))
            weights[nbwgt2-1:0:-1] = -firstfactor * sigma
            weights[nbwgt2+1:-1] = -firstfactor * sigma

        else:
            raise IOError('Unknowm filter %s must be "lp", "hp" or "bp"'
                          % filt_type)

        self.wgt = weights

    def wgt_runave(self, data):

        """ Compute the running mean of a ND input array using the filter weights.

        :param numpy.array data: Array to filter 
         out (time must be the first dimension)

        """

        # we retrive the wgt array and initialise the output
        wgt = self.wgt
        output = np.zeros(data.shape)

        nwt = len(wgt)
        nwgt2 = nwt//2
        indw = nwgt2

        if data.ndim > 1:
            shapein = np.array(data.shape)
            shapein = shapein[::-1]
            shapein[-1] = 1
            wgt = np.tile(wgt, shapein)
            wgt = np.transpose(wgt)

        while indw+nwgt2+1 <= data.shape[0]:
            index = np.arange(indw-nwgt2, indw+nwgt2+1)
            output[indw] = np.sum(wgt*data[index], axis=0)
            indw = indw+1

        output[output == 0] = np.nan

        return output
    
def projection(vn, c=51, m=1, mflg='off', wnx=10, wnxflg='all', time_range=['1979-01-01','2019-12-31'], lat_range=[90, -90], pic_save='./',dataflg=''):
    # zmode: the vertical mode, default m = 1
    # m: wave truncation
    # wnx: zonal wave number truncation [inclusive]
    # time_range: the time range of the data used in training and validating the model [inclusive]
    # lat_range: the latitude range (y) of the data used in projection

    # read data; any file with olr[time, lat, lon]
    if dataflg=='raw':
        fn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/daily/'+vn+'.day.1978to2023.nc'
    elif dataflg=='new':
        fn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/fltano120/'+vn+'.fltano120.1978to2023based1979to2012.nc'
    else:
        fn = '/pscratch/sd/l/linyaoly/ERA5/reanalysis/ERA5.'+vn+'GfltG.day.1901to2020.nc'
    
    ds = xr.open_dataset(fn)

    ds1 = ds.sel(time=slice(time_range[0], time_range[1]), lat=slice(lat_range[0], lat_range[1])).fillna(0)

    olr = ds1[vn].values  # (time, lat, lon)
    lat = ds1['lat']
    lon = ds1['lon'].values
    time = ds1['time'].values

    if mflg=='off':
        olr_re = np.copy(olr)  # no dimension reduction on the meridional direction. 
    else:  
        # # parameters
        # N = 1e-2  # buoyancy frequency (s-1)
        # H = 1.6e4  # tropopause height (m)
        beta= 2.28e-11  # variation of coriolis parameter with latitude
        # g = 9.8  # gravity acceleration 
        # theta0 = 300  # surface potential temperature (K)
        # c = N * H / np.pi / zmode # gravity wave speed
        L = np.sqrt(c / beta)  # horizontal scale (m)
    
        # define y = lat * 110 km / L
        y = lat.values * 110 * 1000 / L # dimensionless

        # define meridianol wave structures
        phi = []

        # define which equatorial wave is included in the reconstructed map
        # m is analogous to a meridional wavenumber
        # m = 0: mostly Kelvin wave
        # m = 2: mostly Rossby wave

        if mflg=='odd':
            m_list = np.arange(1,m,2)  
        elif mflg=='even':
            m_list = np.arange(0,m,2)
        elif mflg=='all':
            m_list = np.arange(m)
        elif mflg=='one':
            m_list = [m-1]
        elif mflg=='1pls':
            m_list = [0,m-1]
        elif mflg=='no1':
            m_list = np.arange(1,m)
        elif mflg=='resi':
            m_list = np.arange(m)  # this is the part to be removed from filtered map
        else:
            print('wrong m flag!')
            exit()

        for i in m_list:
            p = special.hermite(i)
            Hm = p(y)
            phim = np.exp(- y**2 / 2) * Hm / np.sqrt((2**i) * np.sqrt(np.pi) * math.factorial(i))

            phi.append(np.reshape(phim, (1, len(y), 1)))

        # projection coefficients
        olrm = []

        dy = (lat[0].values - lat[1].values) * 110 * 1000 / L 

        for i in range(len(m_list)):
            um = np.sum(olr * phi[i] * dy, axis=1, keepdims=True)  # (time, 1, lon)
            olrm.append(um)

        # reconstruction 
        olr_re = np.zeros(np.shape(olr))  # (time, lat, lon)

        for i in range(len(m_list)):
            olr_re = olr_re + olrm[i] * phi[i]
        
        if mflg=='resi':
            olr_re1 = olr - olr_re
            del olr_re
            olr_re = np.copy(olr_re1)
            del olr_re1

    if wnxflg=='off':
        olr_re_fft = np.copy(olr_re)
    else:
        # do fourier transform along each latitude at each time step
        coef_fft = np.fft.rfft(olr_re, axis=2)
        # remove waves whose zonal wave cycles are larger than wnx
        if wnxflg=='all':
            coef_fft[:,:,wnx+1:] = 0.0 
        elif wnxflg=='one':
            coef_fft[:,:,:wnx] = 0.0
            coef_fft[:,:,wnx+1:] = 0.0 
        elif wnxflg=='no0':  # include 1, 2, ..., wnx
            coef_fft[:,:,wnx+1:] = 0.0 
            coef_fft[:,:,0] = 0.0
        elif wnxflg=='no0p7': # include 1, 2, ..., wnx, 7
            coef_fft[:,:,wnx+1:7] = 0.0 
            coef_fft[:,:,8:] = 0.0 
            coef_fft[:,:,0] = 0.0
        elif wnxflg=='resi':  # resi of 0-wnx[inclusive]
             coef_fft[:,:,:wnx+1] = 0.0
        else:
            print('wrong wnx flag!')
            exit()
        # reconstruct OLR with selected zonal waves
        olr_re_fft = np.fft.irfft(coef_fft, np.shape(olr_re)[2], axis=2)

    if pic_save != 'None':
        fig_file_path = pic_save+dataflg+vn+str(lat_range[0])+'deg_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg+'.jpg'
        if not os.path.exists(fig_file_path):  # create a snapshot of the input map if the file does not exist. 
            # save the OLR maps at the first time step
            fig, ax = plt.subplots(2,1)
            fig.set_figheight(6)
            fig.set_figwidth(12)

            # fig1: reconstructed OLR after zonal fft
            im = ax[0].contourf(lon, lat.values, olr_re_fft[0,:,:])
            ax[0].set_xlabel('longitude')
            ax[0].set_ylabel('latitude')
            plt.colorbar(im, ax=ax[0])
            ax[0].set_title('filtered+Yproj_m'+str(m)+mflg+'_wnx'+str(wnx)+wnxflg)

            # fig2: filtered OLR - reconstructed OLR after zonal fft
            tmp = olr[0,:,:] - olr_re_fft[0,:,:] 
            im = ax[1].contourf(lon, lat.values, tmp)
            ax[1].set_xlabel('longitude')
            ax[1].set_ylabel('latitude')
            plt.colorbar(im, ax=ax[1])
            ax[1].set_title('information missing from the filtered map')

            plt.subplots_adjust(hspace=0.4)

            fig.savefig(fig_file_path)

    return olr_re_fft  # (time, lat, lon)

# write a set of functions to isolate Kelvin wave and Rossby wave signals from data

'''
1. nondimensionalize variables and maintain the first baroclinic mode; get u[time, lat, lon] and theta[time, lat, lon]
2. projection onto parabolic cylinder functions
3. calculate K, R1, R2, ...
4. reconstruct signals with selected signals
'''

def nondim_u(u850, u200, c=51.0):
    # 1. divided by c
    # 2. u = (u850 -u200) / 2 / sqrt(2)

    u = (u850[:] - u200[:]) / 2.0 / np.sqrt(2) / c

    return u  # u[time, lat, lon]

def nondim_theta(Z850, Z200, c=51.0):
    # Z: m2s-2
    # geopotential scale: c2
    # 1. divided by c2
    # 2. theta = (Z200-Z850) / 2 / sqrt(2)

    theta = (Z200[:] - Z850[:]) / 2.0 / np.sqrt(2) / c / c

    return theta  # theta[time, lat, lon]

def get_rl(u, theta):

    r = (u - theta) / np.sqrt(2.0)
    l = (u + theta) / np.sqrt(2.0)

    return r, l

def get_phimy(m, y):
    p = special.hermite(m)
    Hm = p(y)
    phim = np.exp(- y**2 / 2) * Hm / np.sqrt((2**m) * np.sqrt(np.pi) * math.factorial(m))

    return phim 

def proj_on_phi(u, theta, c=51.0, Kelvin_only=True, Rossby_only=False):
    beta = 2.28e-11
    L = np.sqrt(c / beta)
    # u is the input variable u[time, lat, lon]
    kt, klat, klon = np.shape(u)
    y = np.linspace(90,-90,klat) * 110000 / L
    dy = y[0] - y[1]
    
    if Kelvin_only:
        phi0 = np.reshape(get_phimy(0, y), [1, klat, 1])

        u0 = np.sum(u * phi0 * dy, axis=1, keepdims=True)  # axis=1 means sum all latitudes
        theta0 = np.sum(theta * phi0 * dy, axis=1, keepdims=True)  #

        K = (u0 - theta0) / np.sqrt(2)  #K[time, 1, lon]

        u_re = K * phi0 / np.sqrt(2)

        return u_re

