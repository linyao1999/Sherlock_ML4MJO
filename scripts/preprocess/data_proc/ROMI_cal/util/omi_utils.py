import numpy as np
import pandas as pd
import datetime
from numba import jit
import mjoindices.empirical_orthogonal_functions as eof
from util.MJO_indices_util import get_anomalies_1var
# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
@jit(nogil=True)
def runningMeanFast_conv(x, N):
    N2 = np.int_((N-1)/2)
    out = np.convolve(x, np.ones((N,))/N, mode='valid') 
    padbeg = np.zeros((N2,))
    padend = np.zeros((N2,))

    padbeg[0] = x[0]
    for i in np.arange(1,N2):
        padbeg[i] = x[0:2*i+1].mean()

    xrev = x[::-1]
    padend[0] = xrev[0]
    for i in np.arange(1,N2):
        padend[i] = xrev[:2*i+1].mean()
    padend = padend[::-1]

    out = np.concatenate(  ( padbeg, out, padend )  )

    return out
# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
@jit(nogil=True)
def runningMeanFast_conv_wrong(x, N):
    out = np.convolve(x, np.ones((N,))/N,mode='valid')[(N-1):]
    padbeg = np.zeros((N+1,))
    padend = np.zeros((N-3,))

    padbeg[0] = x[0]
    for i in np.arange(1,N+1):
        padbeg[i] = x[i-i:i+i].mean()

    xrev = x[::-1]
    padend[0] = xrev[0]
    for i in np.arange(1,N-3):
        padend[i] = xrev[i-i:i+i].mean()
    padend = padend[::-1]

    out = np.concatenate(  ( padbeg, out, padend )  )

    return out

# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------

def get_omi_eofs(lon,lat, eofnpzfile = '/pscratch/sd/l/linyaoly/ERA5/EOF/EOFs_daily1979to2014_Oct18.npz'):
    # -20 to 20
   eofs = eof.restore_all_eofs_from_npzfile(eofnpzfile)
   eof1_leap = np.zeros((366,lat.size,lon.size))
   eof2_leap = np.zeros((366,lat.size,lon.size))

   for i in np.arange(366):
        eof1tmp = eofs.eof1vector_for_doy(i+1)
        eof1_leap[i,:,:] = eof1tmp.reshape((1,lat.size,lon.size))
        eof2tmp = eofs.eof2vector_for_doy(i+1)
        eof2_leap[i,:,:] = eof2tmp.reshape((1,lat.size,lon.size))
        
   eof1 = eof1_leap[0:365,:,:]
   eof2 = eof2_leap[0:365,:,:]

   mjo_mode = {'eof1':eof1, 'eof2':eof2}

   mjo_mode.update({'eof1_leap':eof1_leap, 'eof2_leap':eof2_leap})
   return mjo_mode



# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
def get_olr4omi(olrDataFile):

    # Access the required variables
    x = get_anomalies_1var('olr', olrDataFile, None, None, latsel=20, yearclim_sta=1979, yearclim_end=2014)  
    x = x.drop('dayofyear')

    lats = x['lat'].values.squeeze()
    lons = x['lon'].values.squeeze()

    nt, nlat, nlon = x.shape

    date_sta_np = x['time'].values[0]
    date_sta = pd.to_datetime(date_sta_np).to_pydatetime()

    ttime = [date_sta + datetime.timedelta(int(tt)) for tt in np.arange(nt)]

    olr_rt_noac = np.squeeze(np.asarray(x.values))
        
    olr_rt_noac_sm = np.zeros(olr_rt_noac.shape)

    for it in np.arange(40, olr_rt_noac.shape[0]):
        olr_rt_noac_sm[it,:,:] = olr_rt_noac[it,:,:] - olr_rt_noac[it-40:it,:,:].mean(0)

    olr_rt_noac_sm_tapavg = np.zeros(olr_rt_noac_sm.shape)
    if 1 == 1:
        for ii in np.arange(olr_rt_noac_sm_tapavg.shape[1]):
          for jj in np.arange(olr_rt_noac_sm_tapavg.shape[2]):
            olr_rt_noac_sm_tapavg[:, ii, jj]  = runningMeanFast_conv(olr_rt_noac_sm[:, ii, jj], 9)

    return olr_rt_noac, olr_rt_noac_sm, olr_rt_noac_sm_tapavg, ttime, lons, lats

# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
def calc_mjo_ind(olr_in, tin, mjo_mode):
    """ 
     olr_in: olr anomaly starting from 20N to 20S (20, 17.5, ..., -20)
     mjo_mode: EOFs from 20N to 20S (20, 17.5, ..., -20)
    """
    mjo_rt = np.zeros((olr_in.shape[0], 2))    
    for it in np.arange(olr_in.shape[0]):
        a1 = olr_in[it,:,:]
        iday = tin[it].timetuple().tm_yday - 1
        
        if (tin[it].year % 4 == 0):
            a2 = mjo_mode['eof1_leap'][iday,:,:].T      
            mjo_rt[it, 0] = np.sum(a1*a2 )

            a2 = mjo_mode['eof2_leap'][iday,:,:].T      
            mjo_rt[it, 1] = np.sum(a1*a2 )
        else:
            a2 = mjo_mode['eof1'][iday,:,:].T      
            mjo_rt[it, 0] = np.sum(a1*a2 )

            a2 = mjo_mode['eof2'][iday, :,:].T      
            mjo_rt[it, 1] = np.sum(a1*a2 )
    return mjo_rt

# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------
def calc_romi(olr_rt_noac_sm_tapavg, ttime, mjo_mode):
    """ 
     olr_rt_noac_sm_tapavg: 9-pt averaged from obs, starting from 20N to 20S (20, 17.5, ..., -20) 
     mjo_mode: EOFs from 20S to 20N (-20, -17.5, ..., 20)
    """ 
    mjo_rt2 = np.zeros((olr_rt_noac_sm_tapavg.shape[0], 2))    
    for it in np.arange(olr_rt_noac_sm_tapavg.shape[0]):
        iday = ttime[it].timetuple().tm_yday - 1
        if (ttime[it].year % 4 == 0):
            a1 = olr_rt_noac_sm_tapavg[it,::-1,:]
        
            a2 = mjo_mode['eof1_leap'][iday,:,:]      
            mjo_rt2[it, 0] = np.sum(a1*a2 )
        
            a2 = mjo_mode['eof2_leap'][iday,:,:]     
            mjo_rt2[it, 1] = np.sum(a1*a2 )
        else:
            a1 = olr_rt_noac_sm_tapavg[it,::-1,:]
        
            a2 = mjo_mode['eof1'][iday,:,:]      
            mjo_rt2[it, 0] = np.sum(a1*a2 )
        
            a2 = mjo_mode['eof2'][iday, :,:]     
            mjo_rt2[it, 1] = np.sum(a1*a2 )
    return mjo_rt2
# -------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------