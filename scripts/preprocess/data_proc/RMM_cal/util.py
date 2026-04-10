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
    print('stdolr: ', stdolr**(1/2))
    del tmp 

    # select training u850, time x lon
    tmp = avu850.sel(time=slice(eof_sta,eof_end))
    stdu850 = tmp.var(dim="time")
    stdu850 = stdu850.mean().values
    avu850nm = avu850 / (stdu850**(1/2))
    print('stdu850: ', stdu850**(1/2))  
    del tmp 

    # select training u200, time x lon
    tmp = avu200.sel(time=slice(eof_sta,eof_end))
    stdu200 = tmp.var(dim="time")
    stdu200 = stdu200.mean().values
    avu200nm = avu200 / (stdu200**(1/2))
    print('stdu200: ', stdu200**(1/2))
    del tmp

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

    # select the period of 1979-2001 to do normalization 
    tmp = PC_RMM_field.sel(time=slice(eof_sta, eof_end)) # [time, mode]
    # PC_RMM_field1 = (PC_RMM_field - tmp.mean(dim='time')) / tmp.std(dim='time') 
    PC_RMM_field1 = PC_RMM_field / tmp.std(dim='time') 

    if os.path.exists('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMM_KIM_daily_1979to2001.nc'):
        os.remove('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMM_KIM_daily_1979to2001.nc')
    
    PC_RMM_field1.name = 'RMM'
    PC_RMM_field1.to_netcdf('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMM_KIM_daily_1979to2001.nc')

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
    print('stdolr: ', stdolr**(1/2))
    del tmp 

    # select training u850, time x lon
    tmp = avu850.sel(time=slice(eof_sta,eof_end))
    stdu850 = tmp.var(dim="time")
    stdu850 = stdu850.mean().values
    avu850nm = avu850 / (stdu850**(1/2))
    print('stdu850: ', stdu850**(1/2))
    del tmp 

    # select training u200, time x lon
    tmp = avu200.sel(time=slice(eof_sta,eof_end))
    stdu200 = tmp.var(dim="time")
    stdu200 = stdu200.mean().values
    avu200nm = avu200 / (stdu200**(1/2))
    print('stdu200: ', stdu200**(1/2))
    del tmp 

    # from https://github.com/fredericferry/MJO_RMM/blob/main/MJO_RMM.ipynb
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
    # # from https://github.com/WillyChap/MJOcast/blob/main/build/lib/MJOcast/utils/ProcessOBS.py
    # # avolrnm[:,-1] = 0

    # # 3. concatenate three fields
    # RMM_field = xr.concat([avolrnm,avu850nm,avu200nm], dim="lon")

    # print('RMM_field shape: ', RMM_field.shape)

    # from eofs.xarray import Eof  
    # # solver = Eof(RMM_field.sel(time=slice(eof_sta,eof_end)), center=True) 
    # solver = Eof(RMM_field.sel(time=slice(eof_sta,eof_end)), center=True) 
    # EOF_RMM_field = solver.eofs(neofs=2)  # [mode, lon]
    # EOF_RMM_field = EOF_RMM_field.transpose()  # [lon, mode]

    # # print variance 
    # print(solver.varianceFraction()[:5] * 100)

    # # eigenvalue1 = solver.eigenvalues(neigs=2)

    # Change the Sign of EOF to be consistent with WH04
    ieof1max, ieof2max = EOF_RMM_field[0:len(avolr.lon),:].argmax(dim="lon")
    lonmaxeof1 = EOF_RMM_field.lon[ieof1max]
    lonmaxeof2 = EOF_RMM_field.lon[ieof2max]

    if (lonmaxeof1 >= 100) & (lonmaxeof1 <= 160) :
        EOF_RMM_field[:,0] = - EOF_RMM_field[:,0]

    if (lonmaxeof2 >= 120) & (lonmaxeof2 <= 220) :
        EOF_RMM_field[:,1] = - EOF_RMM_field[:,1]

    # # EOF_RMM_field, eigenvalue1 = get_EOF1979()

    # # project the whole dataset onto the EOF during training
    # EOF_RMM_field1 = EOF_RMM_field.copy()

    # 
    import os 
    if os.path.exists('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMMeof_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc'):
        os.remove('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMMeof_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')

    EOF_RMM_field1.name = 'EOF'
    EOF_RMM_field1.to_netcdf('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMMeof_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')

    if os.path.exists('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMMfield_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc'):
        os.remove('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMMfield_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')

    RMM_field.name = 'RMM_field'
    RMM_field.to_netcdf('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMMfield_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')

def get_RMM(RMM_field, EOF_RMM_field1, eof_sta, eof_end, flg=''):
    
    # PC_RMM_field = RMM_field.dot(EOF_RMM_field1)  # [time, lon] dot [lon, mode] gives [time, mode]
    PC_RMM_field = RMM_field.dot(EOF_RMM_field1)  # [time, lon] dot [lon, mode] gives [time, mode]
    # PC_RMM_field = (RMM_field - RMM_field.sel(time=slice(eof_sta,eof_end)).mean(dim="time")).dot(EOF_RMM_field1)  # [time, lon] dot [lon, mode] gives [time, mode]

    # select the period of 1979-2001 to do normalization 
    tmp = PC_RMM_field.sel(time=slice(eof_sta, eof_end)) # [time, mode]
    # PC_RMM_field1 = (PC_RMM_field - tmp.mean(dim='time')) / tmp.std(dim='time') 
    PC_RMM_field1 = PC_RMM_field / tmp.std(dim='time') 

    print(tmp.std(dim='time'))

    if os.path.exists('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMM_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc'):
        os.remove('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMM_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')
    
    PC_RMM_field1.name = 'RMM'
    PC_RMM_field1.to_netcdf('/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/rmm/RMM_ERA5_daily_'+eof_sta[:4]+'to'+eof_end[:4]+flg+'.nc')

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

