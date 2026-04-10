import xarray as xr
import numpy as np
import util.omi_utils as omi_utils

# set path
# use data from ERA5 daily OLR
olrfile = '/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/data/daily/olr.day.nc'
# Restrict dataset to the original length for the EOF calculation (Kiladis, 2014).
eofnpzfile = '/pscratch/sd/l/linyaoly/ML_MJO_2024_redo/scripts/data_proc/ROMI_cal/EOFs_daily1979to2012.npz'

olr_rt_noac, olr_rt_noac_sm, olr_rt_noac_sm_tapavg, ttime, olons, olats = omi_utils.get_olr4omi(olrDataFile=olrfile)
mjo_mode = omi_utils.get_omi_eofs(olons,olats, eofnpzfile=eofnpzfile)
mjo_rt2 = omi_utils.calc_romi(olr_rt_noac_sm_tapavg, ttime, mjo_mode)

import pandas as pd
ofac = np.std(mjo_rt2[:40*365,0])  # use the first 40 years to calculate the standard deviation
print('ofac:', ofac)
# ofac: 277.8929592651819 (for ROMI, 1979-2012) 

# ofac = 275.5508864118826 
# print('old ofac:', ofac)

momi_1 = mjo_rt2[:,0]/ofac
momi_2 = mjo_rt2[:,1]/ofac
pha = np.arctan2(-momi_1, momi_2)
momi_time = ttime
momi_rt_phase = np.floor((pha+np.pi)/(np.pi/4)).astype(int)
momi_rt_phase += 1
momi_rt_amp = np.sqrt(momi_1**2 + momi_2**2)
momi_rt_df = pd.DataFrame({'omi1':momi_1, 'omi2':momi_2, 'rmma':momi_rt_amp, 'phase':momi_rt_phase, 'date':ttime, })

momi_rt_df['date'] = pd.to_datetime(momi_rt_df['date'])
momi_rt_df = momi_rt_df.set_index('date', inplace=False)
ds = xr.Dataset.from_dataframe(momi_rt_df)

ds = ds.rename({'date':'time'}) 

ds_sel = ds.sel(time=slice('1979',None))

import os 
ROMI_values = np.empty((len(ds_sel['time']),2))
ROMI_values[:,0] = ds_sel['omi1'].values
ROMI_values[:,1] = ds_sel['omi2'].values

ROMI = xr.DataArray(
                    ROMI_values, 
                    dims=['time','mode'], 
                    coords={'time':ds_sel['time'], 'mode':[0,1]}
                    )

ds_romi = xr.Dataset(
    {'ROMI': ROMI},
    coords={'time':ds_sel['time'], 'mode':[0,1]},
)

fn_romi = './ROMI_ERA5_daily_1979to2012.nc'
if os.path.exists(fn_romi):
    os.remove(fn_romi)
ds_romi.to_netcdf(fn_romi, mode='w')
